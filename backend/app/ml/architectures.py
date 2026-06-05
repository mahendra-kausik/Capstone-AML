"""GCN model definitions (mirror research src/)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree


class StaticGCN(nn.Module):
    def __init__(self, in_channels=165, hidden1=64, hidden2=32, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden2, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        logits = self.classifier(x)
        return logits, x


class EvolveGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru = nn.GRUCell(input_size=in_channels, hidden_size=in_channels)
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
    def __init__(self, in_channels=165, hidden1=64, hidden2=32, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1 = EvolveGCNLayer(in_channels, hidden1)
        self.layer2 = EvolveGCNLayer(hidden1, hidden2)
        self.classifier = nn.Linear(hidden2, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.layer1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.layer2(x, edge_index))
        logits = self.classifier(x)
        return logits, x

    def reset_state(self, device):
        self.layer1.reset_state(device)
        self.layer2.reset_state(device)
