import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Linear, GCNConv, GATConv

from .neuron import STLIF, LIF, reset_net


class STFNorm(nn.Module):
    def __init__(self, in_channels: int, v_threshold: float=1.0):
        super().__init__()
        self.v_threshold = v_threshold
        self.lin = Linear(in_channels, in_channels, bias=True, weight_initializer='glorot')
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, spike: Tensor):
            assert spike.ndim == 3
            T, N, d = spike.shape
            out = spike[-1, :, :]
            
            spike = spike.transpose(0, 1).reshape(N, T*d) # N×T×d
            E = torch.mean(spike, dim=-1, keepdim=True)
            Var = torch.std(spike, dim=-1, keepdim=True)
            
            out = self.v_threshold * (out - E) / (Var + 1e-8)
            return self.lin(out)


class STFNConv(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        backbone: str,
        v_threshold: float,
        **kwargs
    ):
        super().__init__()
        assert backbone in ['gcn', 'gat']
        
        if backbone == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        else:
            self.conv = GATConv(in_channels, out_channels, heads=kwargs["num_heads"], concat=False)
        self.norm = STFNorm(in_channels, out_channels)
        self.snn = STLIF(v_threshold=v_threshold)
    
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()
                
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)


class STFN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hid_channels: int, 
        num_heads: int, T: int, v_threshold: float = 1.0, gain: float = 1.0, backbone: str = 'gcn',
    ):
        super().__init__()
        self.gain = gain
        self.T = T

        if backbone == 'gcn':
            self.conv1 = GCNConv(in_channels, hid_channels)
            self.conv2 = GCNConv(hid_channels, out_channels)
        elif backbone == 'gat':
            self.conv1 = GATConv(in_channels, hid_channels, heads=num_heads)
            self.conv2 = GATConv(hid_channels, out_channels, heads=num_heads)
        else:
            raise NotImplementedError

        self.norm1 = STFNorm(hid_channels, hid_channels)
        self.norm2 = STFNorm(out_channels, out_channels)

        self.snn1 = LIF(v_threshold=v_threshold)
        self.snn2 = LIF(v_threshold=v_threshold)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.reset()
    
    def reset(self):
        self.snn1.reset()
        self.snn2.reset()

    def forward(self, x, edge_index, edge_weight=None):
        spike = []
        embd_l_list, embd_r_list = [], []
        for i in range(self.T):
            embd_l = self.conv1(x, edge_index)
            embd_l_list.append(embd_l)
            embd_l = self.norm1(torch.stack(embd_l_list, dim=0))
            out_spike_l = self.snn1(embd_l)

            embd_r = self.conv2(out_spike_l, edge_index)
            embd_r_list.append(embd_r)
            embd_r = self.norm2(torch.stack(embd_r_list, dim=0))
            out_spike_r = self.snn2(embd_r)
            spike.append(out_spike_r)
        reset_net(self)
        return torch.mean(torch.stack(spike, dim=0), dim=0)
