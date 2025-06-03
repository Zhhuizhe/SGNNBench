import torch
import torch.nn as nn
from torch import Tensor

from sgnnbenchmark.neuron import reset_net
from sgnnbenchmark.nn.conv import SpikingGCNConv


class PoissonEncoder(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        out_spike = torch.rand_like(x).le(x).to(x)
        return out_spike


class SpikingGCN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hid_channels: int, num_layers: int, K: int=2, 
        neuron: str='LIF', T: int=20, v_threshold: float=1.0, v_reset: float=0.0, tau: float=2.0, 
        norm: str=None, enable_res: bool=False, enable_jk: bool=False, **kwargs
    ):
        super().__init__()
        self.enable_res = enable_res
        self.enable_jk = enable_jk
        self.T = T

        # Instantiate the Pre-Linear Layer
        self.encoder = PoissonEncoder()
        self.lin_in = nn.Linear(in_channels, hid_channels)
        self.lin_out = nn.Linear(num_layers*hid_channels if enable_jk else hid_channels, out_channels)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SpikingGCNConv(hid_channels, hid_channels, K, 
                neuron, tau, v_threshold, v_reset, norm=norm, enable_res=enable_res))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.lin_out.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters()
            
        self.cache_clear()

    def cache_clear(self):
        for conv in self.convs:
            conv.cache_clear()

    def forward(self, x, edge_index, edge_weight=None):
        out_spikes = []
        for t in range(self.T):
            embd = self.lin_in(x)
            
            final_embd = []
            for i in range(len(self.convs)):
                embd = self.convs[i](embd, edge_index, edge_weight)
                
                final_embd.append(embd)
            
            if self.enable_jk:
                embd = torch.cat(final_embd, dim=-1)
            out_spikes.append(embd)
        
        out_spikes = torch.stack(out_spikes).mean(dim=0)
        out_spikes = self.lin_out(out_spikes)
        
        reset_net(self)
        return out_spikes
