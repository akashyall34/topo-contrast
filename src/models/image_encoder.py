"""3D image encoder for the pilot.

For the pilot we use a small from-scratch 3D CNN rather than the full
nnU-Net ResEnc-L backbone — it's fast enough to iterate on a laptop/single
GPU and is only meant to answer the go/no-go question. Swap in the real
nnU-Net encoder (`get_network_from_plans` in nnunetv2) once the pilot
passes and full pretraining starts.
"""
import torch
import torch.nn as nn


class PilotImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        c = 16
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, c, 3, stride=2, padding=1), nn.InstanceNorm3d(c), nn.LeakyReLU(inplace=True),
            nn.Conv3d(c, c * 2, 3, stride=2, padding=1), nn.InstanceNorm3d(c * 2), nn.LeakyReLU(inplace=True),
            nn.Conv3d(c * 2, c * 4, 3, stride=2, padding=1), nn.InstanceNorm3d(c * 4), nn.LeakyReLU(inplace=True),
            nn.Conv3d(c * 4, c * 8, 3, stride=2, padding=1), nn.InstanceNorm3d(c * 8), nn.LeakyReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(c * 8, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W) patch centered on a node/branch of interest."""
        feats = self.net(x)
        pooled = self.pool(feats).flatten(1)
        return self.proj(pooled)
