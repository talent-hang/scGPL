import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv


class GATLayer(nn.Module):
    """封装 PyTorch Geometric 的 GNN 实现"""

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2):
        super(GATLayer, self).__init__()
        self.out_features = out_features
        out_channels = out_features // num_heads
        self.conv = GATConv(
            in_channels=in_features,
            out_channels=out_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            bias=True,
        )
        self.fallback = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return self.fallback(h)
        return self.conv(h, edge_index)


class GCNLayer(nn.Module):
    """标准 GCN 卷积"""

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels=in_features, out_channels=out_features)
        self.fallback = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return self.fallback(h)
        return self.conv(h, edge_index)


class GraphSAGELayer(nn.Module):
    """GraphSAGE 卷积"""

    def __init__(self, in_features, out_features):
        super(GraphSAGELayer, self).__init__()
        self.conv = SAGEConv(in_channels=in_features, out_channels=out_features, normalize=True)
        self.fallback = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return self.fallback(h)
        return self.conv(h, edge_index)


class GINLayer(nn.Module):
    """GIN 卷积，内部使用两层 MLP"""

    def __init__(self, in_features, out_features):
        super(GINLayer, self).__init__()
        hidden = max(out_features, in_features)
        mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
        self.conv = GINConv(mlp)
        self.fallback = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return self.fallback(h)
        return self.conv(h, edge_index)


def build_gnn_layer(layer_type: str, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.2):
    layer_type = layer_type.lower()
    if layer_type == 'gat':
        return GATLayer(in_features, out_features, num_heads=num_heads, dropout=dropout)
    if layer_type == 'gcn':
        return GCNLayer(in_features, out_features)
    if layer_type == 'graphsage':
        return GraphSAGELayer(in_features, out_features)
    if layer_type == 'gin':
        return GINLayer(in_features, out_features)
    raise ValueError(f"Unsupported gnn layer type: {layer_type}")

