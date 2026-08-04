"""DEPRECATED — superseded design, kept for reference only, not wired into
any current entrypoint. See docs/PROJECT.md Section 1. Current pretraining
uses src/train_connectivity_pretrain.py instead.

Pilot / full contrastive pretraining loop.

Pilot mode (`configs/pilot.yaml`) runs this on ~30-50 volumes for a few
epochs, then hands the trained image encoder to
`src/probes/branch_identity_probe.py` to decide whether the full approach
is worth pursuing. Full mode (`configs/full_run.yaml`) runs the same loop
at scale.
"""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.contrastive_dataset import ContrastiveVesselDataset, collate_single_case
from src.losses.contrastive_loss import info_nce
from src.models.contrastive_head import ProjectionHead
from src.models.graph_encoder import GraphEncoder
from src.models.image_encoder import PilotImageEncoder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_models(cfg: dict, device: torch.device):
    image_encoder = PilotImageEncoder(embed_dim=cfg["embed_dim"]).to(device)
    graph_encoder = GraphEncoder(embed_dim=cfg["embed_dim"]).to(device)
    image_head = ProjectionHead(in_dim=cfg["embed_dim"], out_dim=cfg["proj_dim"]).to(device)
    graph_head = ProjectionHead(in_dim=cfg["embed_dim"], out_dim=cfg["proj_dim"]).to(device)
    return image_encoder, graph_encoder, image_head, graph_head


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ContrastiveVesselDataset(Path(cfg["data_dir"]), patch_size=cfg["patch_size"])
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single_case)

    image_encoder, graph_encoder, image_head, graph_head = build_models(cfg, device)
    params = (
        list(image_encoder.parameters()) + list(graph_encoder.parameters())
        + list(image_head.parameters()) + list(graph_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"])

    ckpt_dir = Path(cfg["output_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            patches = batch["patches"].to(device)
            node_feats = batch["node_feats"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_feats = batch["edge_feats"].to(device)

            if patches.shape[0] < 2:
                continue  # InfoNCE needs >=2 nodes to have negatives

            img_embed = image_head(image_encoder(patches))
            graph_embed_full = graph_encoder(node_feats, edge_index, edge_feats)
            graph_embed = graph_head(graph_embed_full)

            loss = info_nce(img_embed, graph_embed, temperature=cfg["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{cfg['epochs']}  loss={avg_loss:.4f}")

        torch.save(
            {"image_encoder": image_encoder.state_dict(), "graph_encoder": graph_encoder.state_dict()},
            ckpt_dir / "latest.pth",
        )

    return image_encoder, graph_encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
