import sys

import torch
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix

from .neuron import reset_net, LIF, IF, PLIF
sys.path.append("../")
from utils import RandomWalkSampler, Sampler, add_selfloops


class SAGEAggregator(nn.Module):
    def __init__(self, in_features, out_features,
                 aggr='mean',
                 concat=False,
                 bias=False):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.aggr = aggr
        self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]

        self.lin_l = nn.Linear(in_features, out_features, bias=bias)
        self.lin_r = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, neigh_x):
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x, dim=0)

        if not isinstance(neigh_x, torch.Tensor):
            neigh_x = torch.cat([self.aggregator(h, dim=1)
                                for h in neigh_x], dim=0)
        else:
            neigh_x = self.aggregator(neigh_x, dim=1)

        x = self.lin_l(x)
        neigh_x = self.lin_r(neigh_x)
        out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggr={self.aggr})"


class SpikeNet(nn.Module):
    def __init__(
        self, data, T, hids=[128, 64], sizes=[5, 2], alpha=1.0,
        dropout=0.7, act='LIF', aggr='mean', sampler='sage',
        surrogate='triangle', concat=False, bias=True,
    ):
        super().__init__()

        in_features, out_features = data.num_features, data.num_classes
        
        tau = 1.0
        if sampler == 'rw':
            self.sampler = RandomWalkSampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
        elif sampler == 'sage':
            self.sampler = Sampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
        else:
            raise ValueError(sampler)

        del data.edge_index

        aggregators, snn = nn.ModuleList(), nn.ModuleList()

        for hid in hids:
            aggregators.append(SAGEAggregator(in_features, hid,
                                              concat=concat, bias=bias,
                                              aggr=aggr))

            if act == "IF":
                snn.append(IF(alpha=alpha, surrogate=surrogate))
            elif act == 'LIF':
                snn.append(LIF(tau, alpha=alpha, surrogate=surrogate))
            elif act == 'PLIF':
                snn.append(PLIF(tau, alpha=alpha, surrogate=surrogate))
            else:
                raise ValueError(act)

            in_features = hid * 2 if concat else hid

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.snn = snn
        self.sizes = sizes
        self.T = T
        self.pooling = nn.Linear(T * in_features, out_features)

    def encode(self, x, nodes):
        spikes = []
        sizes = self.sizes
        device = x.device

        for time_step in range(self.T):
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes

            for size in sizes:
                nbr = self.sampler(nbr, size)
                num_nodes.append(nbr.size(0))
                h.append(x[nbr].to(device))

            for i, aggregator in enumerate(self.aggregators):
                self_x = h[:-1]
                neigh_x = []
                for j, n_x in enumerate(h[1:]):
                    neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))

                out = self.snn[i](aggregator(self_x, neigh_x))
                if i != len(sizes) - 1:
                    out = self.dropout(out)
                    h = torch.split(out, num_nodes[:-(i + 1)])

            spikes.append(out)
        spikes = torch.cat(spikes, dim=1)
        reset_net(self)
        return spikes

    def forward(self, x, nodes):
        spikes = self.encode(x, nodes)
        return self.pooling(spikes)