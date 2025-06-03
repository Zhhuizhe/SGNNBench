import torch
import torch.nn as nn
from torch_geometric.utils import spmm

from sgnnbenchmark.neuron import IF, LIF, PLIF, reset_net


class SpikingGCN(nn.Module):
    def __init__(
        self, data, in_channels: int, out_channels: int, hid_channels: int, K: int, 
        neuron: str='LIF', v_threshold: float=1.0, tau: float=1.0, surrogate: float='triangle', 
        dropout: float=0.7, enabel_norm: bool=False
    ):
        super().__init__()
        self.K = K
        self.dropout = dropout
        self.enable_norm = enabel_norm
        
        self.lin_proj = nn.Linear(in_channels, hid_channels)
        self.lin_out = nn.Linear(len(data)*hid_channels, out_channels)
        self.norm = nn.BatchNorm1d(hid_channels)
        
        if neuron == "IF":
            self.snn = IF(v_threshold, surrogate=surrogate)
        elif neuron == 'LIF':
            self.snn = LIF(tau, v_threshold, surrogate=surrogate)
        elif neuron == 'PLIF':
            self.snn = PLIF(tau, v_threshold, surrogate=surrogate)
        else:
            raise ValueError(neuron)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        self.lin_out.reset_parameters()
        self.norm.reset_parameters()

    def encode(self, data, nodes, device, adj):
        spikes = []
        for t in range(len(data)):
            snapshot, edge_index = data[t], adj[t]
            x, edge_index = snapshot.x.to(device), edge_index.to(device)
            
            embd = self.lin_proj(x)
            embd = spmm(edge_index, embd)
            # del edge_index
            if self.enable_norm:
                self.norm(embd)
            out = self.snn(embd[nodes])
            spikes.append(out)
        spikes = torch.cat(spikes, dim=1)
        reset_net(self)
        return spikes

    def forward(self, data, nodes, device, adj):
        spikes = self.encode(data, nodes, device, adj)
        return self.lin_out(spikes)
