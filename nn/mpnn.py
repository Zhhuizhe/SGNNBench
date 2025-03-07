import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, 
    GATConv, 
    SAGEConv, 
    SGConv
)


class MPNN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hid_channels: int, 
        num_layers: int=2, dropout: float=0.0, conv_type: str="gcn", **kwargs,
    ):
        super().__init__()
        assert conv_type in ["gcn", "gat", "sage", "sgc"]
        self.dropout = dropout
        
        channels = [in_channels] + [hid_channels]*num_layers + [out_channels]
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for i in range(len(channels)-1):
            if conv_type == "gcn":
                self.convs.append(GCNConv(channels[i], channels[i+1]))
            elif conv_type == "gat":
                self.convs.append(GATConv(channels[i], channels[i+1], kwargs["num_heads"], concat=False))
            elif conv_type == "sage":
                self.convs.append(SAGEConv(channels[i], channels[i+1], normalize=kwargs["normalize"]))
            elif conv_type == "sgc":
                self.convs.append(SGConv(channels[i], channels[i+1], kwargs["K"]))
            self.norms.append(nn.BatchNorm1d(channels[i+1]))
        
        self.reset_parameters()   

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.norms[i](x).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x