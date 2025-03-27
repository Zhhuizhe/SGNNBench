import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Linear, GCNConv, GATConv

from .neuron import STLIF, LIF, reset_net


class STFNorm(nn.BatchNorm1d):
    def __init__(
        self, 
        num_features: int, 
        v_th: float,
        eps: float=1e-05,
        momentum: float=0.1,
        affine: bool=True,
        track_running_stats: bool=True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.num_features = num_features
        self.v_th = v_th
        self.running_mean = torch.reshape(self.running_mean, (1, 1, -1))
        self.running_var = torch.reshape(self.running_var, (1, 1, -1))
    
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
            
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean(dim=(0, 2), keepdim=True)
            # use biased var in train
            var = input.var(dim=(0, 2), unbiased=False, keepdim=True) # compute variance via the biased estimator (i.e. unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                num_nodes, num_features = input.size(1), input.size(2)
                self.running_mean = self.running_mean.expand(1, num_nodes, num_features)
                self.running_var = self.running_var.expand(1, num_nodes, num_features)

                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        mean, var = mean.squeeze(0), var.squeeze(0)
        input = self.v_th * (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine: # if True, we use the affine transformation (linear transformation)
            input = input * self.weight[None, None, :] + self.bias[None, None, :]

        return input[-1, :, :]
      

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

        self.norm1 = STFNorm(hid_channels, v_threshold)
        self.norm2 = STFNorm(out_channels, v_threshold)

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
