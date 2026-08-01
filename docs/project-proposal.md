# Project Proposal: Image–Topology Contrastive Pretraining for Tubular Anatomy Segmentation

> TODO: paste in the full proposal draft (motivation, related work, method
> details, pilot success criteria, target venue/timeline). Referenced from
> the README's Overview section and by `configs/pilot.yaml`'s
> `min_accuracy_gain` threshold, which should trace back to a rationale
> written here.

## Pilot gate

`scripts/run_pilot.sh` trains the contrastive pipeline on a ~30-50 volume
ImageCAS subset and runs `src/probes/branch_identity_probe.py`, which
compares linear-probe branch-identity accuracy for the contrastively
pretrained image encoder vs. a randomly initialized baseline. The decision
threshold lives in `configs/pilot.yaml` (`min_accuracy_gain`) — fill in the
justified value here before running for real.
