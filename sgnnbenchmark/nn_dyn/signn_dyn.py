import torch
import torch.nn as nn

from sgnnbenchmark.utils import RandomWalkSampler, Sampler, add_selfloops
from sgnnbenchmark.neuron import BLIF, reset_net


class TAlayer(nn.Module):
    def __init__(
        self, in_features, hids=[128, 64], sizes=[5, 2],
        v_threshold=1.0, alpha=1.0, surrogate='triangle', 
        concat=False, bias=False, aggr='mean', dropout=0.5
    ):
        super().__init__()

        self.Agg_1 = Aggregator(in_features, hids[0], concat=concat, bias=bias, aggr=aggr)
        self.Agg_2 = Aggregator(hids[0], hids[1], concat=concat, bias=bias, aggr=aggr)
        
        self.snn_1 = BLIF(hid=hids[0], v_threshold=v_threshold, alpha=alpha, surrogate=surrogate)
        self.snn_2 = BLIF(hid=hids[1], v_threshold=v_threshold, alpha=alpha, surrogate=surrogate)
        
        self.dropout = nn.Dropout(dropout)
        self.sizes = sizes
    
    def reset_parameters(self):
        self.Agg_1.reset_parameters()
        self.Agg_2.reset_parameters()

    def forward(self, h, num_nodes):
        for i in range(len(self.sizes)):
            self_x = h[:-1]
            neigh_x = []
            for j, n_x in enumerate(h[1:]):
                neigh_x.append(n_x.view(-1, self.sizes[j], h[0].size(-1)))
            if i != len(self.sizes) - 1:
                out_t, out_x = self.Agg_1(self_x, neigh_x)
                out_t = self.snn_1(out_t)
                out_s = torch.mul(out_x, out_t)
                out_s = self.dropout(out_s)
                h = torch.split(out_s, num_nodes[:-(i + 1)])
            else:
                out_t, out_x = self.Agg_2(self_x, neigh_x)
                out_t = self.snn_2(out_t)
                out = torch.mul(out_x, out_t)
                out = self.dropout(out)
        return out
        
               
class Aggregator(nn.Module):
    def __init__(self, in_features: int, out_features: int, aggr: str='mean', concat: bool=False, bias: bool=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.aggr = aggr
        self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]

        self.lin_l = nn.Linear(in_features, out_features, bias=bias)
        self.lin_r = nn.Linear(in_features, out_features, bias=bias)
        
        self.lin_l_t = nn.Linear(in_features, out_features, bias=bias)
        self.lin_r_t = nn.Linear(in_features, out_features, bias=bias)
        
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, x, neigh):
        if not isinstance(neigh, torch.Tensor):
            neigh_h = []
            for i in range(len(neigh)):
                h = self.aggregator(neigh[i], dim=1)
                neigh_h.append(h)
            neigh = torch.cat(neigh_h, dim=0)
             
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x, dim=0)
        
        self_x = self.lin_l(x)
        self_s = self.lin_l_t(x)
        neigh_x = self.lin_r(neigh)
        neigh_s = self.lin_r_t(neigh)
        
        out_t =  self_s + neigh_s
        out_x =  self.act(self_x + neigh_x)

        return out_t, out_x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggr={self.aggr})"


class SiGNN(nn.Module):
    def __init__(
        self, data, in_features, out_features, hids=[128, 64], sizes=[5, 2], 
        alpha=1.0, p=0.5, dropout=0.7, bias=True, aggr='mean', sampler='sage',
        surrogate='triangle', concat=False, nchannels=3, invth=1
    ):
        super().__init__()

        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj_evolve]
        elif sampler == 'sage':
            self.sampler = [Sampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [Sampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj_evolve]
        else:
            raise ValueError(sampler)

        TA_layers = nn.ModuleList()
        for i in range(nchannels):
            TA_layers.append(TAlayer(in_features, hids=hids, sizes=sizes, v_threshold=invth, 
                                     alpha=alpha, surrogate=surrogate, concat=concat, bias=bias, aggr=aggr, dropout=dropout))


        num_steps = len(data)
        self.TA_layers = TA_layers
        self.sizes = sizes
        self.p = p
        self.MTGagg = nn.Linear(hids[-1], out_features)
        self.pooling_1 = nn.Conv1d(groups=hids[-1], in_channels=hids[-1], out_channels=hids[-1], kernel_size=num_steps)
        self.pooling_2 = nn.Conv1d(groups=hids[-1], in_channels=hids[-1], out_channels=hids[-1], kernel_size=(num_steps//2 + num_steps%2))
        self.pooling_3 = nn.Conv1d(groups=hids[-1], in_channels=hids[-1], out_channels=hids[-1], kernel_size=(num_steps//3 + num_steps%3))
    
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
                
        for layer in self.TA_layers:
            layer.reset_parameters()

    def encode(self, data, nodes, device):
        embeddings1 = []
        embeddings2 = []
        embeddings3 = []
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

            if time_step % 1 == 0:
                o1 = self.TA_layers[0](h, num_nodes)
                embeddings1.append(o1)
            
            if time_step % 2 == 0:
                o2 = self.TA_layers[1](h, num_nodes)
                embeddings2.append(o2)

            if (time_step + 1) % 3 == 0 and len(data)==27:  
                o3 = self.TA_layers[2](h, num_nodes)           
                embeddings3.append(o3)
            elif time_step % 3 == 0 and len(data)!=27:
                o3 = self.TA_layers[2](h, num_nodes)            
                embeddings3.append(o3)
            
        emb1 = torch.stack(embeddings1)
        emb1 = emb1.permute(1, 2, 0) 
        emb1 = self.pooling_1(emb1).squeeze(dim=2)
        
        emb2 = torch.stack(embeddings2)
        emb2 = emb2.permute(1, 2, 0) 
        emb2 = self.pooling_2(emb2).squeeze(dim=2)
        
        emb3 = torch.stack(embeddings3)
        emb3 = emb3.permute(1, 2, 0) 
        emb3 = self.pooling_3(emb3).squeeze(dim=2)

        embeddings = torch.stack([emb1, emb2, emb3], dim=0)
        embeddings = torch.mean(embeddings, dim=0)
        embeddings = self.MTGagg(embeddings)
        
        reset_net(self)
        return embeddings

    def forward(self, data, nodes, device, adj):
        embeddings = self.encode(data, nodes, device)
        return embeddings