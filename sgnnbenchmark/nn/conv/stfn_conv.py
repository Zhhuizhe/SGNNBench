import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import spmm
from torch_geometric.typing import SparseTensor, OptTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax, spmm

from spikingjelly.clock_driven.neuron import LIFNode, ParametricLIFNode


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
        # Dynamically calculate the mean and var for different time steps
        self._cached_input.append(input)
        membrane_potential = torch.stack(self._cached_input) # T×N×d
        
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            mean = membrane_potential.mean(dim=(0, 2), keepdim=True) # T×N×d -> 1×N×1
            # use biased var in train
            var = membrane_potential.var(dim=(0, 2), unbiased=False, keepdim=True) # compute variance via the biased estimator (i.e. unbiased=False)
            n = membrane_potential.numel() / membrane_potential.size(1)
            with torch.no_grad():
                mean = mean.squeeze(0).repeat(1, input.size(1)) # 1×N×1 -> N×d
                var = var.squeeze(0).repeat(1, input.size(1))
                
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # mean, var = mean.squeeze(-1), var.squeeze(-1)
        input = self.rho * self.v_th * (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine: # if True, we use the affine transformation (linear transformation)
            input = input * self.weight[None, :] + self.bias[None, :]
            
        return input


class GCConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr: str='mean', **kwargs):
        super().__init__(aggr, **kwargs)
        self.proj_weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.proj_bias = nn.Parameter(torch.empty(1, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_weight)
        nn.init.zeros_(self.proj_bias)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # if x.dim() != 3:
        #     raise ValueError(f"expected 3D input (got {x.dim()}D input)")
        x = F.linear(x, self.proj_weight)
        
        row, col = edge_index
        deg_inv_sqrt = degree(col, x.size(0), dtype=x.dtype).pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col] # calculate the c_{ij}
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out
        
    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j * edge_weight.view(-1, 1) + self.proj_bias
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr) 
    

class GAConv(MessagePassing):
    def __init__(
        self, in_channels:int, out_channels: int, num_heads: int,
        negative_slope: float=0.2, dropout: float=0.0, aggr='mean', **kwargs
    ):
        super().__init__(aggr, **kwargs)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.proj_weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.proj_out = nn.Parameter(torch.empty(out_channels, out_channels))
        
        self.att_src = nn.Parameter(torch.empty(1, num_heads, out_channels//num_heads))
        self.att_dst = nn.Parameter(torch.empty(1, num_heads, out_channels//num_heads))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_weight)
        nn.init.xavier_uniform_(self.proj_out)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        N, H, C = x.size(0), self.num_heads, self.out_channels//self.num_heads
        x = F.linear(x, self.proj_weight).view(N, H, C)
        
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)
        
        alpha = self.edge_updater(edge_index, alpha=alpha)

        x, alpha = x.transpose(0, 1), alpha.transpose(0, 1) # N×H×d' -> H×N×d'
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = F.linear(out.transpose(0, 1).reshape(N, H*C), self.proj_out) # H×N×d' -> N×H×d' -> N×d
        return out

    def edge_update(self, alpha_j: Tensor, alpha_i: Tensor, index: Tensor, ptr: OptTensor) -> Tensor:
        alpha = alpha_i + alpha_j
        if index.numel() == 0:
            return alpha
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
        
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j


class STFNConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, backbone: str,
        neuron: str='LIF', v_threshold: float=1.0, num_heads: int=None, 
        norm: str='bn', enable_res: bool=False
    ):
        super().__init__()
        self.norm = norm
        self.enable_res = enable_res
        
        self.lin_res = nn.Linear(in_channels, out_channels)
        
        if backbone == 'gcn':
            self.conv = GCConv(in_channels, out_channels)
        elif backbone == 'gat':
            self.conv = GAConv(in_channels, out_channels, num_heads)
        else:
            raise NotImplementedError
        
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'stfn':
            self.norm = STFNorm(out_channels, v_threshold)
        else:
            raise NotImplementedError(f'Only BatchNormalization (bn) and STFNormalization (stfn) are implemented. ({norm})')
        
        if neuron == 'LIF':
            self.snn = LIFNode(v_threshold=v_threshold)
        elif neuron == 'PLIF':
            self.snn = ParametricLIFNode(v_threshold=v_threshold)
        else:
            raise NotImplementedError(neuron)
    
        self.reset_parameters()
    
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()
    
    def reset(self):
        self.snn.reset()
                
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_res = self.lin_res(x)
        x = self.conv(x, edge_index)
        
        # Residual Part
        if self.enable_res:
            x = self.norm(x) + x_res
        else:
            x = self.norm(x)
        
        x = self.snn(x)
        return x