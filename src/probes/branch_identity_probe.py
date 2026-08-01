"""The pilot's go/no-go gate.

Question: does contrastive alignment against the skeleton graph teach the
image encoder to distinguish *which branch* a patch belongs to, better
than an unaligned (randomly initialized / autoencoder-pretrained) image
encoder does?

Method: freeze the image encoder, extract patch embeddings for every node
in a held-out set of cases, and train a small linear/kNN classifier to
predict branch identity (a discrete label: e.g. "LAD", "LCx", "RCA", or
just a per-case bifurcation-cluster id if anatomical labels aren't
available). Compare probe accuracy: contrastive-pretrained encoder vs.
baseline encoder.

Decision rule (fill in your actual threshold before running):
    contrastive_probe_acc - baseline_probe_acc >= cfg["min_accuracy_gain"]
    => go: full pretraining pipeline is worth building.
    else => no-go, or revisit the contrastive formulation before scaling.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader

from src.datasets.contrastive_dataset import ContrastiveVesselDataset, collate_single_case
from src.models.image_encoder import PilotImageEncoder


@torch.no_grad()
def extract_embeddings(encoder: PilotImageEncoder, loader: DataLoader, device: torch.device):
    """Returns (embeddings, branch_labels) across all cases.

    Branch labels here default to a per-case, per-connected-component id
    (i.e. "which major branch of this specific vessel tree"), since that's
    derivable purely from the graph without anatomical annotation. Swap in
    real LAD/LCx/RCA labels if/when you have them — that's a strictly
    harder and more meaningful version of the same probe.
    """
    encoder.eval()
    all_embeds, all_labels = [], []
    for batch in loader:
        patches = batch["patches"].to(device)
        if patches.shape[0] < 2:
            continue
        embeds = encoder(patches).cpu().numpy()
        labels = branch_labels_from_graph(batch)
        all_embeds.append(embeds)
        all_labels.append(labels)
    return np.concatenate(all_embeds), np.concatenate(all_labels)


def branch_labels_from_graph(batch) -> np.ndarray:
    """Cluster graph nodes into branches via connected components after
    removing the highest-degree (root/bifurcation-heavy) nodes — a coarse
    but annotation-free proxy for "which branch". Replace with real
    anatomical labels if available."""
    import networkx as nx

    edge_index = batch["edge_index"].numpy()
    n_nodes = batch["node_feats"].shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edge_index.T.tolist())

    degrees = dict(g.degree())
    high_degree_nodes = {n for n, d in degrees.items() if d >= 3}
    g.remove_nodes_from(high_degree_nodes)

    labels = np.full(n_nodes, -1, dtype=int)
    for comp_id, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            labels[n] = comp_id
    for n in high_degree_nodes:
        labels[n] = -2  # bifurcation nodes get their own class
    return labels


def run_probe(embeddings: np.ndarray, labels: np.ndarray, cv_folds: int = 5) -> float:
    mask = labels >= -2  # keep all; drop this line if you want to exclude bifurcation nodes
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    scores = cross_val_score(clf, embeddings[mask], labels[mask], cv=cv_folds)
    return float(scores.mean())


def main(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ContrastiveVesselDataset(Path(cfg["data_dir"]), patch_size=cfg["patch_size"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single_case)

    # Contrastive-pretrained encoder
    contrastive_encoder = PilotImageEncoder(embed_dim=cfg["embed_dim"]).to(device)
    ckpt = torch.load(Path(cfg["output_dir"]) / "latest.pth", map_location=device)
    contrastive_encoder.load_state_dict(ckpt["image_encoder"])

    # Baseline: same architecture, random init, never trained
    baseline_encoder = PilotImageEncoder(embed_dim=cfg["embed_dim"]).to(device)

    contrastive_embeds, labels = extract_embeddings(contrastive_encoder, loader, device)
    baseline_embeds, _ = extract_embeddings(baseline_encoder, loader, device)

    contrastive_acc = run_probe(contrastive_embeds, labels)
    baseline_acc = run_probe(baseline_embeds, labels)
    gain = contrastive_acc - baseline_acc

    print(f"baseline probe accuracy:     {baseline_acc:.4f}")
    print(f"contrastive probe accuracy:  {contrastive_acc:.4f}")
    print(f"gain:                        {gain:+.4f}")

    threshold = cfg.get("min_accuracy_gain", 0.10)
    decision = "GO" if gain >= threshold else "NO-GO"
    print(f"decision (threshold={threshold}): {decision}")
    return {"baseline_acc": baseline_acc, "contrastive_acc": contrastive_acc, "gain": gain, "decision": decision}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
