# Image–Topology Contrastive Pretraining for Tubular Anatomy Segmentation

**Target venue:** MICCAI (main track); MIDL as fallback/alternate
**Type:** Solo research project, single-GPU (AWS g6e.xlarge, L40S), ~$200 compute budget

---

## 1. Problem Statement

Self-supervised pretraining for 3D medical image segmentation has matured substantially (AMAES, HySparK, SwinMM, AdvMIM, VIS-MAE), but all of these methods operate purely in image space — masking, reconstructing, or contrasting voxel patches against each other. None incorporate the *explicit topological structure* of the anatomy being imaged, even for structures — vascular trees, airway trees — where that structure is well-defined, cheaply extractable from raw intensity via offline morphological processing (Hessian/Frangi filtering, 3D skeletonization), and known to be diagnostically load-bearing (branch connectivity, bifurcation geometry).

Separately, graph-based methods for tubular anatomy exist but serve different purposes: GLCP (MICCAI 2025, Oral) is a supervised segmentation architecture that predicts skeleton/discontinuity maps to refine segmentation, with no pretraining stage and no explicit graph representation. VesselGPT (MICCAI 2025) tokenizes vessel geometry and autoregressively generates novel vascular trees — a generative synthesis method operating on graphs alone, with no image encoder. Neither learns a joint image-to-topology representation.

The closest true prior art is Khanna et al. (2024), "Learning Generalized Medical Image Representations through Image-Graph Contrastive Pretraining," which contrasts chest X-rays against knowledge graphs extracted from radiology reports. The mechanism (image encoder + graph encoder, contrastive alignment) is structurally similar. The graph modality is not: their graphs encode clinical/textual relations; ours would encode physical anatomical topology (bifurcation nodes, branch edges, geometric attributes).

**Honest novelty scope (revised after review — two prior framings were rejected):**

1. *Rejected: image patch contrastively aligned to the graph node extracted from that same patch/volume.* The graph is a deterministic function of the same intensity data (Frangi + skeletonize) — the "positive pair" carries no information the image encoder couldn't already extract via reconstruction alone. Not a real information-injection mechanism; indistinguishable in kind from predicting a hand-crafted feature (HOG-style) of the same input.
2. *Rejected: cross-patient contrastive alignment by canonical topological role* (e.g., "second-order branch, 15mm from ostium" as a positive-pair label shared across patients). Collides with existing work — SAM, "Self-supervised Learning of Pixel-wise Anatomical Embeddings" (Yan et al., arxiv 2012.02383), already does same-anatomical-position-across-images contrastive learning generically. Also rests on a cross-patient anatomical homology assumption (coronary dominance patterns, missing branches, disease-altered topology) that is fragile precisely in diseased vessels — the case that matters clinically.
3. **Adopted:** predict the *topological connectivity relationship* between two spatially separated, non-overlapping image crops from the *same* volume (same-branch / common-ancestor-within-k-hops / disconnected, or continuous geodesic tree-distance). The graph is used only to generate this pair label at preprocessing time — it is not a parallel encoded modality and no graph encoder is required at train or inference time. See "Pretraining architecture" below.

**The gap:** no published method pretrains an image encoder by requiring it to predict anatomical-graph-derived connectivity between disjoint image regions, as opposed to (a) reconstructing/masking pixels, (b) contrasting an image against a redundant self-derived feature, or (c) matching canonical anatomical position across independent scans.

---

## 2. Proposed Method

**Offline preprocessing (per volume, CPU, one-time cost):**
1. Multi-scale Hessian/Frangi vesselness filtering on raw intensity (no ground-truth labels used, to avoid pretraining-time label leakage)
2. 3D morphological skeletonization (`skimage.morphology.skeletonize_3d`) to extract centerlines
3. Bifurcation/junction detection (skeleton voxels with >2 26-connected neighbors)
4. Construction of anatomical graph G = (V, E): nodes = bifurcation points + sampled centerline points, edges = branch segments with geometric attributes (length, tortuosity, radius estimate)
5. Serialize to `.npz` per volume

**Pretraining architecture — topological-connectivity prediction:**
- Image encoder: 3D CNN/ViT (reuse nnU-Net ResEnc backbone for downstream compatibility), shared weights, siamese over a pair of crops
- No graph encoder in the trained path. The vessel graph (Section 2, offline preprocessing) is used only to look up the topological relationship between two crops' nearest graph nodes and produce a label; it is discarded after label generation. This keeps the deployed/fine-tuned model a plain image encoder — no auxiliary graph module dependency at inference, unlike Khanna-style or GLCP architectures.
- Relational head: lightweight MLP over the concatenated (or difference) embeddings of the two crops, predicting the connectivity class (same-branch / common-ancestor-within-k-hops / disconnected) or regressing geodesic tree-distance
- Crop-pair sampling, per volume:
  - Standard random pairs (non-overlapping, spaced beyond the encoder's receptive field)
  - **Hard negatives:** spatially close, topologically disconnected (e.g. a diagonal branch running near-parallel to the LAD)
  - **Hard positives:** spatially distant, topologically connected (two points far apart along one long, tortuous branch)
  - Hard-example mining is not optional — see "Shortcut-learning safeguard" below

**Shortcut-learning safeguard (mandatory, run before any full build):**
Topological connectivity correlates with Euclidean proximity (vessels are spatially continuous curves), so naive random-pair sampling lets the model solve the pretext task via implicit distance/location estimation rather than genuine anatomical reasoning — the same failure mode documented for Rubik's-Cube/jigsaw relative-position pretext tasks (shortcut through absolute coordinates/scan geometry, not content). Required falsification check: train a baseline that sees *only* the Euclidean distance between crop centers (no image content) and predicts connectivity from that scalar alone. If this baseline matches the real model's accuracy, the task is a distance regression in disguise and must be discarded or re-designed (denser hard-example mining) before proceeding. This check is cheap (half a day) and must run before the encoder/relational-head architecture work below.

**Why this task, mechanistically:** segmentation methods with limited receptive fields are known to produce false merges/splits — exactly what Betti number error (Δβ₀, Δβ₁) measures. This pretext task is designed to teach long-range connectivity reasoning directly. The resulting hypothesis is falsifiable and specific: pretraining on connectivity prediction should reduce Betti errors more than it reduces plain Dice. If ablations show a uniform small bump across all metrics rather than concentration on topological metrics, that is evidence of noise, not mechanism, and should be reported as such.

**Downstream fine-tuning:**
- Load pretrained image encoder into nnU-Net
- Combined loss: weighted BCE + Dice + Skeleton Recall Loss (ECCV 2024 formulation)
- Standard 5-fold cross-validation

---

## 3. Datasets

| Dataset | Anatomy | Size | Role |
|---|---|---|---|
| ImageCAS | Coronary CTA | ~1000 volumes | Primary — pretraining + fine-tuning |
| ASOCA | Coronary CTA | ~40 volumes | Secondary validation / small-data regime test |
| PARSE2022 | Pulmonary airway CT | ~100 volumes | Cross-anatomy generalization check (stretch goal, only if time permits) |

ImageCAS as primary addresses the sample-size risk that applies to ASOCA-only pretraining (n=40 is too small to expect a self-supervised method to show a reliable gain over baselines).

---

## 4. Baselines (required, not optional)

1. **Random init** — no pretraining, sanity floor
2. **Image-only 3D MAE** — standard masked reconstruction, isolates whether the connectivity-prediction signal adds anything beyond generic self-supervision
3. **Rubik's-Cube / relative-position prediction** (Zhuang et al.-style 3D pretext task) — required, not optional. Isolates the actual claim: does *anatomically-grounded* relational prediction (same-branch/disconnected) beat *raw geometric* relational prediction (relative 3D offset)? Without this baseline, the paper cannot distinguish its mechanism from a known, already-published pretext-task family.
4. **GLCP** — strong supervised-only segmentation baseline, shows pretraining benefit over a non-pretrained SOTA method
5. **LA-CAF** (MICCAI 2025) — CLIP-based vision-language segmentation for pulmonary artery/vein, validated on PARSE2022 with published DSC numbers; a different mechanism (vision-language fusion, not topology graphs) but a strong SOTA supervised benchmark on overlapping anatomy/dataset

**Related work / rejected alternatives, addressed but not run as baselines (one-line rationale each, so this doesn't need re-deriving during rebuttal):**
- *Khanna et al. image-graph contrastive* — structurally similar mechanism, but their graph modality (clinical text) is independent information; ours (same-volume topology graph) is not, which is exactly the flaw the current design was chosen to avoid. Cite as motivating prior art, not as a required empirical baseline.
- *SAM (pixel-wise anatomical embeddings, arxiv 2012.02383)* — cross-image same-position contrastive learning; different task shape (cross-instance position matching vs. within-volume relational connectivity) and not subject to our cross-patient homology concern. Cite as related work.

*Note: an earlier draft of this baseline list referenced a method named "Spark3D" as a 3D MAE comparison. That name did not resolve to a verifiable paper on search — it may be a garbled reference to HySparK ("Hybrid Sparse Masking for Large-Scale Medical Image Pre-Training"). Verify the exact method and citation before it goes near a related-work section; an unverifiable citation is a fast way to lose reviewer trust.*

---

## 5. Ablations

- With/without hard-example mining in crop-pair sampling (isolates whether the shortcut-learning safeguard is load-bearing, not decorative)
- Connectivity-label task vs. Rubik's-Cube/relative-position task, same architecture and compute (isolates anatomically-grounded relation from raw geometric relation — the crux ablation for this paper's actual claim)
- Connectivity label granularity: binary connected/disconnected vs. fine-grained hop-distance / geodesic-distance regression
- Downstream loss: with/without Skeleton Recall Loss (isolates pretraining gain from loss-function gain)
- Data-efficiency curve: fine-tune on 10% / 25% / 50% / 100% of labeled data — the strongest plot for a pretraining paper, directly demonstrates annotation-burden reduction
- Metric concentration check: does the pretraining gain concentrate on topological metrics (Betti error, clDice) vs. diffuse evenly across all metrics (Dice, NSD)? A uniform small bump everywhere reads as noise, not mechanism, and should be reported as a negative result if that's what's observed

---

## 6. Evaluation Metrics

- Dice, clDice (centerline Dice — topology-sensitive)
- Betti number errors (Δβ₀ — connected-component/fragmentation error; Δβ₁ — loop/handle error) — stronger than clDice alone for demonstrating connectivity preservation specifically, standard in this literature
- Normalized Surface Dice (NSD) — boundary agreement within a distance tolerance
- Data efficiency (metric vs. % labeled data used across 10%/25%/50%/100% splits)

---

## 7. Compute Budget

AWS g6e.xlarge (L40S, spot, ~$1.17/hr) — offline preprocessing done locally/CPU to conserve budget for GPU-bound pretraining and 5-fold fine-tuning runs. FP8/bfloat16 mixed precision, gradient checkpointing for memory headroom.

---

## 8. Risk / Open Questions (to resolve before full build)

The pilot gate is now a three-step sequence, each cheaper and more decisive
than the last. Do not proceed to the next step until the current one passes —
each is designed to kill the project quickly and inexpensively if the core
bet is wrong, rather than discovering that after a full build.

1. **Sampling feasibility check (do this first, on 5–10 volumes, no training):** at the crop size/spacing chosen for pretraining, verify that hard-negative pairs (spatially close, topologically disconnected) and hard-positive pairs (spatially distant, topologically connected) actually exist in usable quantity. If the crop size is too large or too small relative to typical branch spacing, one of these pools will be empty or degenerate, and the task collapses before it starts. Estimated cost: under an hour, CPU only.

2. **Shortcut-learning falsification check (mandatory gate, before any encoder training):** train a baseline that predicts connectivity from *only* the Euclidean distance between crop centers (no image content). **Pass condition:** the real siamese-encoder model must beat this baseline by a real margin on held-out pairs. If it doesn't, the task is a distance-regression task in disguise and must be redesigned (denser/harder mining) or abandoned — the full build is not worth starting on a task a one-line distance function already solves. Estimated cost: under $10, CPU/single GPU, a few hours.

3. **Primary empirical risk — does the connectivity-prediction signal produce a measurable downstream gain over both image-only MAE and Rubik's-Cube/relative-position pretraining at this dataset scale?** Train on a small subset (~30–50 ImageCAS volumes), 15–20 epochs, reduced crop size. Freeze the encoder and train a linear probe to classify anatomical hierarchy level (e.g., primary trunk vs. secondary vs. tertiary branch) from patch embeddings. **Pass condition:** probe accuracy meaningfully above the Rubik's-Cube-pretrained baseline's probe accuracy — not chance, not random init. Beating chance is close to guaranteed regardless of mechanism (branch level correlates with visible vessel caliber) and does not answer the actual question. If connectivity-prediction pretraining doesn't beat the Rubik's-Cube baseline here, the "anatomically-grounded relation beats raw geometric relation" bet is dead and the full build is not worth starting. Estimated cost: under $10, 4–8 GPU-hours.

4. Full-text read of Khanna et al. (currently only abstract-level verified) and of SAM (arxiv 2012.02383), to confirm the differentiation arguments in Section 1 hold up before they go into a related-work section.
5. Confirm no pretraining-time label leakage — vesselness/skeleton primitives must derive strictly from raw intensity, never ground-truth masks. This holds unchanged under the new design: the graph is still derived only from Frangi + skeletonize output, now consumed as a pair-label generator rather than a contrastive target.