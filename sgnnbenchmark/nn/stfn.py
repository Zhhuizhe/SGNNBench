import torch
import torch.nn as nn
from torch import Tensor

from sgnnbenchmark.nn.conv.stfn_conv import STFNConv
from sgnnbenchmark.neuron import reset_net


class STFN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hid_channels: int, num_layers: int,
        T: int, neuron: str='LIF', v_threshold: float = 1.0, backbone: str = 'gcn', num_heads: int=None, 
        norm: str='stfn', enable_res: bool=False, enable_jk: bool=False, **kwargs
    ):
        super().__init__()
        self.T = T
        self.enable_jk = enable_jk
        
        # Classification Head
        self.lin_out = nn.Linear(hid_channels if not enable_jk else num_layers*hid_channels, out_channels)
        
        channels = [in_channels] + [hid_channels]*num_layers
        self.convs = nn.ModuleList()
        for i in range(0, len(channels)-1):
            self.convs.append(
                STFNConv(channels[i], channels[i+1], backbone, neuron, v_threshold, num_heads, norm, enable_res)
            )

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_out.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x: Tensor, edge_index :Tensor) -> Tensor:
        spike_out = list()
        for t in range(self.T):
            spike_embd = x
            final_embd = []
            for conv in self.convs:
                spike_embd = conv(spike_embd, edge_index)
                final_embd.append(spike_embd)
            
            # For jumping knowledge, concatenating spiking embeddings from each layer
            if self.enable_jk:
                spike_embd = torch.cat(final_embd, dim=-1)
            
            spike_out.append(spike_embd)
        
        print(torch.stack(spike_out).detach().mean())
        # Classification Head
        spike_out = torch.stack(spike_out).mean(dim=0)
        spike_out = self.lin_out(spike_out)
        reset_net(self)
        return spike_out
