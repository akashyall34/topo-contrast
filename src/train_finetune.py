"""Fine-tunes a segmentation model from a contrastively pretrained image
encoder. Only meant to run after the pilot gate (branch_identity_probe)
reports GO and full pretraining has produced outputs/full_pretrain/latest.pth.

This is deliberately minimal (Dice+CE loss, no skeleton-recall term yet —
see src/losses/skeleton_recall_loss.py) since it's downstream of the pilot
decision and shouldn't be over-built before that decision is made.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.datasets.segmentation_dataset import SegmentationDataset
from src.models.image_encoder import PilotImageEncoder


class SimpleSegHead(nn.Module):
    """Minimal decoder pairing with PilotImageEncoder's 4x-downsampled
    feature map; swap for the real nnU-Net decoder once this is the
    full-run pipeline rather than a pilot follow-on."""

    def __init__(self, embed_dim: int, out_channels: int = 1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 32, 4, stride=4),
            nn.InstanceNorm3d(32), nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, 4, stride=4),
            nn.InstanceNorm3d(16), nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, out_channels, 1),
        )

    def forward(self, x):
        return self.up(x)


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return 1 - (2 * intersection + eps) / (denom + eps)


def train(cfg: dict, fraction: float = 1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(Path(cfg["finetune_data_dir"]), fraction=fraction)
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 2), shuffle=True)

    encoder = PilotImageEncoder(embed_dim=cfg["embed_dim"]).to(device)
    pretrain_ckpt = Path(cfg["output_dir"]) / "latest.pth"
    if pretrain_ckpt.exists():
        encoder.load_state_dict(torch.load(pretrain_ckpt, map_location=device)["image_encoder"])
    else:
        print(f"warning: no pretrained checkpoint at {pretrain_ckpt}, training from scratch")

    head = SimpleSegHead(embed_dim=cfg["embed_dim"]).to(device)
    bce = nn.BCEWithLogitsLoss()
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.get("finetune_lr", 1e-4))

    for epoch in range(cfg.get("finetune_epochs", 100)):
        epoch_loss = 0.0
        for batch in loader:
            image, mask = batch["image"].to(device), batch["mask"].to(device)
            feats = encoder.net(image)  # skip the pooled proj, keep spatial map
            logits = head(feats)
            logits = nn.functional.interpolate(logits, size=mask.shape[2:], mode="trilinear", align_corners=False)

            loss = bce(logits, mask) + dice_loss(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"epoch {epoch + 1}  loss={epoch_loss / max(len(loader), 1):.4f}")

    return encoder, head


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, fraction=args.fraction)
