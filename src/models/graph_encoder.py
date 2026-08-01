"""Graph attention encoder over the vessel bifurcation graph.

Produces one embedding per node (bifurcation/endpoint), to be contrasted
against an image-patch embedding centered on that same node's voxel
coordinates.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim: int = 4, edge_in_dim: int = 5, hidden_dim: int = 64,
                 embed_dim: int = 128, num_layers: int = 3, heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_in_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """x: (N, node_in_dim), edge_index: (2, E), edge_attr: (E, edge_in_dim).
        Returns per-node embeddings (N, embed_dim)."""
        h = self.input_proj(x)
        for conv, norm in zip(self.layers, self.norms):
            h = h + torch.relu(conv(h, edge_index, edge_attr))
            h = norm(h)
        return self.out_proj(h)
