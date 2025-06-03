import torch
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix

from sgnnbenchmark.neuron import reset_net, LIF, IF, PLIF
from sgnnbenchmark.utils import RandomWalkSampler, Sampler, add_selfloops


class SAGEAggregator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggr: str='mean'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggr = aggr
        self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]

        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)

    def forward(self, x, neigh_x):
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x, dim=0)

        if not isinstance(neigh_x, torch.Tensor):
            neigh_x = torch.cat([self.aggregator(h, dim=1) for h in neigh_x], dim=0)
        else:
            neigh_x = self.aggregator(neigh_x, dim=1)

        x = self.lin_l(x)
        neigh_x = self.lin_r(neigh_x)
        out = x + neigh_x
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, aggr={self.aggr})"


class SpikeNet(nn.Module):
    def __init__(
        self, data, T: int, hids=[128, 64], sizes=[5, 2], 
        alpha: float=1.0, dropout: float=0.7, act: float='LIF', aggr: float='mean', 
        sampler: float='sage', surrogate: float='triangle', enable_pre_lin: bool=False
    ):
        super().__init__()
        in_channels, out_channels = data.num_features, data.num_classes
        self.enable_pre_lin = enable_pre_lin
        self.sizes = sizes
        self.T = T
        
        if enable_pre_lin:
            self.lin_in = nn.Linear(in_channels, hids[0])
        else:
            self.register_parameter('lin_in', None)
        
        tau = 1.0
        if sampler == 'rw':
            self.sampler = RandomWalkSampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
        elif sampler == 'sage':
            self.sampler = Sampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
        else:
            raise ValueError(sampler)

        self.aggregators, self.snn = nn.ModuleList(), nn.ModuleList()
        for hid in hids:
            self.aggregators.append(SAGEAggregator(
                in_channels if not enable_pre_lin else hids[0], hid, aggr=aggr
            ))

            if act == "IF":
                self.snn.append(IF(alpha=alpha, surrogate=surrogate))
            elif act == 'LIF':
                self.snn.append(LIF(tau, alpha=alpha, surrogate=surrogate))
            elif act == 'PLIF':
                self.snn.append(PLIF(tau, alpha=alpha, surrogate=surrogate))
            else:
                raise ValueError(act)

            in_channels = hid

        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.Linear(T * in_channels, out_channels)

    def encode(self, x, nodes):
        spikes = []
        sizes = self.sizes
        device = x.device

        if self.enable_pre_lin:
            x = self.lin_in(x)
        
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