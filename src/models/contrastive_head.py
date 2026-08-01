"""Projection heads mapping image/graph embeddings into a shared contrastive space."""
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
