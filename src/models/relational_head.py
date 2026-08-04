"""Relational head predicting topological connectivity between two crop
embeddings (docs/PROJECT.md Section 2). Consumes embeddings from a shared
siamese image encoder (src/models/image_encoder.py::PilotImageEncoder) —
no graph encoder anywhere in this path.
"""
import torch
import torch.nn as nn


class RelationalHead(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embed_a: torch.Tensor, embed_b: torch.Tensor) -> torch.Tensor:
        """embed_a, embed_b: (B, embed_dim). Returns (B, num_classes) logits.

        The true label is symmetric in (a, b) (connectivity doesn't depend
        on which crop was sampled first), but [emb_a, emb_b] concatenation
        alone isn't — swapping the inputs gives a different feature vector
        unless the MLP happens to learn that symmetry from data. Appending
        |a-b| (which *is* symmetric) gives the head a head start on that,
        it doesn't enforce it structurally.
        """
        diff = torch.abs(embed_a - embed_b)
        return self.net(torch.cat([embed_a, embed_b, diff], dim=-1))
