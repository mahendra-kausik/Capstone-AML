"""EvolveGCN-H model (Pareja et al. -H variant) for disjoint temporal graphs."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


class EvolveGCNLayer(nn.Module):
    """GRU-evolved weight matrix + manual normalized graph convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru = nn.GRUCell(input_size=in_channels, hidden_size=in_channels)
        # Match GCNConv (bias=True); without bias, evolved weights drive all-negative
        # pre-activations and ReLU zeroes every node embedding.
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.W_state = None

    def reset_state(self, device: torch.device):
        w = torch.empty(self.out_channels, self.in_channels, device=device)
        nn.init.xavier_uniform_(w)
        self.W_state = w

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.W_state is None:
            self.reset_state(x.device)
        if self.W_state.device != x.device:
            self.W_state = self.W_state.to(x.device)

        w_in = self.W_state.detach()
        # EvolveGCN-H: evolve the weight matrix using each row as GRU input/hidden.
        # Do NOT feed x.mean() to every row — identical inputs collapse all W rows
        # across snapshots (rank-1 weights) before the first training step finishes.
        w_evolved = self.gru(w_in, w_in)
        self.W_state = w_evolved.detach()

        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index_sl
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        agg = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(
            0,
            col.unsqueeze(1).expand(-1, x.size(1)),
            norm.unsqueeze(1) * x[row],
        )
        return F.linear(agg, w_evolved, self.bias)


class EvolveGCNH(nn.Module):
    """165 → hidden1 → hidden2 → 2 with configurable dropout."""

    def __init__(
        self,
        in_channels: int = 165,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.layer1 = EvolveGCNLayer(in_channels, hidden1)
        self.layer2 = EvolveGCNLayer(hidden1, hidden2)
        self.classifier = nn.Linear(hidden2, 2)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, edge_index)
        x = F.relu(x)
        embeddings = x
        logits = self.classifier(x)
        return logits, embeddings

    def reset_state(self, device):
        self.layer1.reset_state(device)
        self.layer2.reset_state(device)

    def reset_weights(self):
        for layer in (self.layer1, self.layer2):
            for name, param in layer.gru.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def get_state(self):
        return (
            self.layer1.W_state.clone() if self.layer1.W_state is not None else None,
            self.layer2.W_state.clone() if self.layer2.W_state is not None else None,
        )

    def set_state(self, state):
        dev = next(self.parameters()).device
        self.layer1.W_state = state[0].to(dev) if state[0] is not None else None
        self.layer2.W_state = state[1].to(dev) if state[1] is not None else None
