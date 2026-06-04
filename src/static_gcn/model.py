"""Static 2-layer GCN (Elliptic / Upbit shared architecture)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class StaticGCN(nn.Module):
    """165 → 64 → 32 → 2; fixed weights after training."""

    def __init__(
        self,
        in_channels: int = 165,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden2, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        embeddings = x
        logits = self.classifier(x)
        return logits, embeddings
