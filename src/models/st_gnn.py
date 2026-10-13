"""
ResistNet-ST: Simplified Spatio-Temporal GNN
GCN (spatial) -> GRU (temporal) -> Dense
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class STGNN(nn.Module):
    def __init__(self, n_features, gnn_hidden=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(n_features, gnn_hidden)
        self.gcn2 = GCNConv(gnn_hidden, gnn_hidden)
        self.gru = nn.GRU(gnn_hidden, lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: (n_nodes, T, F)
        Returns: (n_nodes, T) logits per timestep
        """
        n_nodes, T, F_dim = x.shape

        # Spatial mixing per timestep
        outputs = []
        for t in range(T):
            h = F.relu(self.gcn1(x[:, t, :], edge_index, edge_weight))
            h = F.relu(self.gcn2(h, edge_index, edge_weight))
            outputs.append(h)
        h_seq = torch.stack(outputs, dim=1)  # (n_nodes, T, gnn_hidden)

        # GRU per node
        gru_out, _ = self.gru(h_seq)  # (n_nodes, T, lstm_hidden)
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out).squeeze(-1)  # (n_nodes, T)
        return logits