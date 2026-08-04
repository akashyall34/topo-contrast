# Project Proposal: Image–Topology Contrastive Pretraining for Tubular Anatomy Segmentation

> TODO: paste in the full proposal draft (motivation, related work, method
> details, pilot success criteria, target venue/timeline). Referenced from
> the README's Overview section and by `configs/pilot.yaml`'s
> `min_accuracy_gain` threshold, which should trace back to a rationale
> written here.

## Pilot gate

Superseded design (see `docs/PROJECT.md` Section 1 for why the original
image-node contrastive framing was dropped: the graph node is a deterministic
function of the same patch/volume, so it injects no new information). The
pretraining task is now topological-connectivity prediction between disjoint
crops from the same volume — the graph is used only to generate pair labels,
not as a contrastive target or a parallel encoder.

The pilot gate is now a three-step sequence (full detail in `docs/PROJECT.md`
Section 8), each gating the next:

1. **Sampling feasibility check** — confirm hard-negative (spatially close,
   topologically disconnected) and hard-positive (spatially distant,
   topologically connected) crop pairs exist in usable quantity at the
   chosen crop size, before any training.
2. **Shortcut-learning falsification check** — a Euclidean-distance-only
   baseline (no image content) must NOT match the real model's accuracy on
   the connectivity-prediction task, or the task is a distance regression
   in disguise and must be redesigned.
3. **Probe comparison** — `src/probes/branch_identity_probe.py` will need
   to change from probe-vs-random-init to probe-vs-Rubik's-Cube-pretrained-
   baseline (chance/random-init is not a real comparison here: branch level
   correlates with visible vessel caliber, so a probe reliably beats chance
   regardless of pretraining mechanism). The decision threshold lives in
   `configs/pilot.yaml` (`min_accuracy_gain`) and should be re-justified
   against the Rubik's-Cube baseline, not against chance, before running for
   real.

**Code status:** `src/` currently implements the superseded image-node
contrastive design end-to-end (`src/datasets/contrastive_dataset.py`,
`src/models/graph_encoder.py`, `src/models/contrastive_head.py`,
`src/losses/contrastive_loss.py`, `src/probes/branch_identity_probe.py`).
None of this has been updated yet for the connectivity-prediction design —
that is a separate, follow-up implementation pass, not covered by this
documentation update.
