from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sgnnbenchmark.neuron import IF, LIF, PLIF, reset_net
from sgnnbenchmark.utils import Sampler, add_selfloops
from .conv import SAGEConv


class STFNorm(nn.BatchNorm1d):
    """
        Implementation of spatial-temporal feature normalization. Link to related paper: https://arxiv.org/abs/2107.06865. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm1d
        eps (float): same with nn.BatchNorm1d
        momentum (float): same with nn.BatchNorm1d
        rho (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm1d
        track_running_stats (bool): same with nn.BatchNorm1d
    """
    def __init__(
        self, num_features: int, v_th: float, eps: float=1e-05, rho: float=1.0,
        momentum: float=0.1, affine: bool=True, track_running_stats: bool=True,
        device=None, dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.rho = rho
        self.v_th = v_th
        
        self._cached_input = []
    
    def reset(self):
        self._cached_input = []
    
    def forward(self, input: Tensor) -> Tensor:
        # self._check_input_dim(input)
        
        # Dynamically calculate the mean and var for different time steps
        self._cached_input.append(input)
        membrane_potential = torch.stack(self._cached_input) # T×N×d

        mean = membrane_potential.mean(dim=(0, 2), keepdim=True) # T×N×d -> 1×N×1
        var = membrane_potential.var(dim=(0, 2), unbiased=False, keepdim=True) # compute variance via the biased estimator (i.e. unbiased=False)
        
        mean = mean.squeeze(0).repeat(1, input.size(1)) # 1×N×1 -> N×d
        var = var.squeeze(0).repeat(1, input.size(1))
            
        input = self.rho * self.v_th * (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine: # if True, we use the affine transformation (linear transformation)
            input = input * self.weight[None, :] + self.bias[None, :]
            
        return input


class STFN(nn.Module):
    def __init__(
        self, data, in_channels, out_channels, hids=[64, 32], sizes=[5, 2],
        neuron='LIF', v_threshold: float=1.0, p=0.5, tau=1.0, dropout=0.7, surrogate='triangle'
    ):
        super().__init__()
        self.dropout = dropout
        self.sizes = sizes
        self.p = p
        
        self.sampler = [Sampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj]
        self.sampler_t = [Sampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj_evolve]

        # if isinstance(hids, List):
        #     channels = [in_channels] + hids
        # else:
        #     raise TypeError('The type of input hidden channels should be "List".')
        channels = [in_channels] + list(hids)
        self.convs, self.neurons, self.norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(channels)-1):
            self.convs.append(SAGEConv(channels[i], channels[i+1]))
            self.norms.append(STFNorm(channels[i+1], v_threshold))
            
            if neuron == "IF":
                self.neurons.append(IF(v_threshold, surrogate=surrogate))
            elif neuron == 'LIF':
                self.neurons.append(LIF(tau, v_threshold, surrogate=surrogate))
            elif neuron == 'PLIF':
                self.neurons.append(PLIF(tau, v_threshold, surrogate=surrogate))
            else:
                raise ValueError(neuron)
        
        # Pooling layer
        self.lin_out = nn.Linear(len(data)*channels[-1], out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_out.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for norm in self.norms:
            norm.reset_parameters()

    def encode(self, data, nodes, device):
        spikes = []
    
        for t in range(len(data)):
            # Neighbor Sampling Phase
            snapshot = data[t]
            sampler, sampler_t = self.sampler[t], self.sampler_t[t]

            x = snapshot.x
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            for size in self.sizes:
                size_1 = max(int(size * self.p), 1)
                size_2 = size - size_1

                if size_2 > 0:
                    nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
                    nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
                    nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
                else:
                    nbr = sampler(nbr, size_1).view(-1)
                # 采样得到二阶领域节点index和对应特征
                num_nodes.append(nbr.size(0))
                h.append(x[nbr].to(device))

            # Message Passing Phase
            for i, mpnn in enumerate(self.convs):
                self_x = h[:-1] # 这里之所以是-1，是因为不只是节点本身需要做消息聚合，第一阶节点也需要消息聚合
                neigh_x = []
                for j, n_x in enumerate(h[1:]):
                    neigh_x.append(n_x.view(-1, self.sizes[j], h[0].size(-1)))

                out = mpnn(self_x, neigh_x)
                out = self.norms[i](out) # STFNorm
                out = self.neurons[i](out) # Spiking Layer
                
                if i != len(self.sizes) - 1:
                    out = F.dropout(out, self.dropout, training=self.training)
                    h = torch.split(out, num_nodes[:-(i + 1)])
            spikes.append(out)
        spikes = torch.cat(spikes, dim=1)
        reset_net(self)
        return spikes

    def forward(self, data, nodes, device, adj):
        spikes = self.encode(data, nodes, device)
        return self.lin_out(spikes)