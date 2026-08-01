"""InfoNCE loss aligning image-patch embeddings with graph-node embeddings.

Positive pair: image patch centered on node i <-> graph embedding of node i
(same volume). Negatives: all other nodes in the batch, including nodes
from other volumes.
"""
import torch
import torch.nn.functional as F


def info_nce(image_embed: torch.Tensor, graph_embed: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """image_embed, graph_embed: (N, D), row i of each is a positive pair.
    Returns symmetric InfoNCE loss (image->graph and graph->image)."""
    image_embed = F.normalize(image_embed, dim=-1)
    graph_embed = F.normalize(graph_embed, dim=-1)

    logits = image_embed @ graph_embed.T / temperature  # (N, N)
    targets = torch.arange(logits.shape[0], device=logits.device)

    loss_i2g = F.cross_entropy(logits, targets)
    loss_g2i = F.cross_entropy(logits.T, targets)
    return (loss_i2g + loss_g2i) / 2
