from typing import Optional

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor

from greatx.nn.layers import PLIF, LIF
from greatx.nn.layers.gcn_conv import make_gcn_norm, make_self_loops
from greatx.functional import spmm

from sgnnbenchmark.nn.conv import STFNorm


class SpikingGCNConv(nn.Module):
    _cached_x: Optional[Tensor]

    def __init__(
        self, in_channels: int, out_channels: int, K: int=1, neuron: str='LIF',
        tau: float=1.0, v_threshold: float=1.0, v_reset: float=0., cached: bool=False,
        add_self_loops: bool=True, normalize: bool=True, norm: str='bn', enable_res: bool=False
    ):
        super().__init__()
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.enable_res = enable_res
        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, weight_initializer='glorot')
        self.lin_res = Linear(in_channels, out_channels, weight_initializer='glorot')
        
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'stfn':
            self.norm = STFNorm(out_channels, v_threshold)
        else:
            raise NotImplementedError(norm)
        
        if neuron == 'LIF':
            self.snn = LIF(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        elif neuron == 'PLIF':
            self.snn = PLIF(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        else:
            raise NotImplementedError(neuron)

        self.reset_parameters()

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
        
        self.cache_clear()

    def cache_clear(self):
        self._cached_x = None
        return self

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        cache = self._cached_x

        if self.enable_res:
            x_res = self.lin_res(x)
        
        if cache is None:
            if self.add_self_loops:
                edge_index, edge_weight = make_self_loops(
                    edge_index, edge_weight, num_nodes=x.size(0))

            if self.normalize:
                edge_index, edge_weight = make_gcn_norm(
                    edge_index, edge_weight, num_nodes=x.size(0),
                    dtype=x.dtype, add_self_loops=False)

            for _ in range(self.K):
                x = spmm(x, edge_index, edge_weight)

            if self.cached:
                self._cached_x = x
        else:
            x = cache.detach()
            
        x = self.lin(x)
        
        if self.enable_res:
            x = self.norm(x) + x_res
        else:
            x = self.norm(x)
        
        x = self.snn(x)
        return x