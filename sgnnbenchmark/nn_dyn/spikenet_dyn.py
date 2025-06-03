import torch
import torch.nn as nn

from sgnnbenchmark.neuron import LIF, IF, PLIF, reset_net
from sgnnbenchmark.utils import RandomWalkSampler, Sampler, add_selfloops
from .conv import SAGEConv


class SpikeNet(nn.Module):
    def __init__(
        self, data, in_features, out_features, hids=[512, 10], sizes=[5, 2], 
        neuron='LIF', surrogate='triangle', alpha=1.0, p=0.5, dropout=0.7, 
        sampler='sage', concat=False, 
    ):
        super().__init__()

        tau = 1.0
        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(
                add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(
                adj_matrix)) for adj_matrix in data.adj_evolve]
        elif sampler == 'sage':
            self.sampler = [Sampler(add_selfloops(adj_matrix))
                            for adj_matrix in data.adj]
            self.sampler_t = [Sampler(add_selfloops(adj_matrix))
                              for adj_matrix in data.adj_evolve]
        else:
            raise ValueError(sampler)

        aggregators, snn = nn.ModuleList(), nn.ModuleList()

        for hid in hids:
            aggregators.append(SAGEConv(in_features, hid, concat))

            if neuron == "IF":
                snn.append(IF(alpha=alpha, surrogate=surrogate))
            elif neuron == 'LIF':
                snn.append(LIF(tau, alpha=alpha, surrogate=surrogate))
            elif neuron == 'PLIF':
                snn.append(PLIF(tau, alpha=alpha, surrogate=surrogate))
            else:
                raise ValueError(neuron)

            in_features = hid * 2 if concat else hid

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.snn = snn
        self.sizes = sizes
        self.p = p
        self.pooling = nn.Linear(len(data) * in_features, out_features)

    def reset_parameters(self):
        self.pooling.reset_parameters()
        
        for aggregator in self.aggregators:
            aggregator.reset_parameters()

    def encode(self, data, nodes, device):
        spikes = []
        sizes = self.sizes
        for time_step in range(len(data)):

            snapshot = data[time_step]
            sampler = self.sampler[time_step]
            sampler_t = self.sampler_t[time_step]

            x = snapshot.x
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            for size in sizes:
                size_1 = max(int(size * self.p), 1)
                size_2 = size - size_1

                if size_2 > 0:
                    nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
                    nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
                    nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
                else:
                    nbr = sampler(nbr, size_1).view(-1)

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

    def forward(self, data, nodes, device, adj=None):
        spikes = self.encode(data, nodes, device)
        return self.pooling(spikes)