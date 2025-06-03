import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from sgnnbenchmark.neuron import reset_net


class Erode(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class MS_MLP_Conv(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int=None, out_channels: int=None, spike_mode: str='LIF'):
        super().__init__()
        out_channels = out_channels or in_channels
        self.hid_channels = hid_channels or in_channels
        self.res = (in_channels == hid_channels)
        
        self.fc1_conv =nn.Linear(in_channels, self.hid_channels)
        self.fc1_bn = nn.BatchNorm1d(self.hid_channels)
        if spike_mode == 'LIF':
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.fc1_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')
        self.fc2_conv =nn.Linear(self.hid_channels, out_channels)


        self.fc2_bn = nn.BatchNorm1d(out_channels)
        if spike_mode == 'LIF':
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.fc2_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x: Tensor) -> Tensor:
        T, B, D = x.shape
        identity = x

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x)
        x=x.reshape(T, B, self.hid_channels).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, D).contiguous()

        x = x + identity
        return x

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()


class MS_SSA_Conv(nn.Module):
    def __init__(self, in_channels: int, num_heads: int=8, spike_mode: int='LIF', dvs: bool=False):
        if spike_mode not in ['LIF', 'PLIF']:
            raise NotImplementedError
        super().__init__()
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125

        self.q_conv = nn.Linear(in_channels, in_channels*self.num_heads)
        self.q_bn = nn.BatchNorm1d(in_channels*self.num_heads)
        if spike_mode == 'LIF':
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.q_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')
        
        self.k_conv = nn.Linear(in_channels, in_channels*self.num_heads)
        self.k_bn = nn.BatchNorm1d(in_channels*self.num_heads)
        if spike_mode == 'LIF':
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.k_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')
        
        self.v_conv = nn.Linear(in_channels, in_channels*self.num_heads)
        self.v_bn = nn.BatchNorm1d(in_channels*self.num_heads)
        if spike_mode == 'LIF':
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.v_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')

        if spike_mode == 'LIF':
            self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.attn_lif = MultiStepParametricLIFNode(init_tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        if spike_mode == 'LIF':
            self.talking_heads_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.talking_heads_lif = MultiStepParametricLIFNode(init_tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')
        self.proj_conv = nn.Linear(in_channels*self.num_heads, in_channels)
        self.proj_bn = nn.BatchNorm1d(in_channels)

        if spike_mode == 'LIF':
            self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.shortcut_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, D = x.shape
        identity = x
        x = self.shortcut_lif(x)
        x_for_qkv = x.flatten(0, 1)  # T*B,D
        q_conv_out = self.q_conv(x_for_qkv)  # T*B,D* self.num_heads
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = q_conv_out.reshape(T, B, D, self.num_heads).permute(0, 1, 3, 2).contiguous()  # T, B,self.num_heads, D
        q = self.q_lif(q_conv_out)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = k_conv_out.reshape(T, B, D, self.num_heads).permute(0, 1, 3, 2).contiguous()  # T, B,self.num_heads, D
        k = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        
        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = v_conv_out.reshape(T, B, D, self.num_heads).permute(0, 1, 3, 2).contiguous()  # T, B,self.num_heads, D
        v = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        
        kv = k.mul(v)
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)

        x = x.reshape(T, B, D * self.num_heads).contiguous()  # T, B,self.num_heads* D
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, D).contiguous() # T, B, D

        x = x + identity
        return x, v

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()


class MS_Block_Conv(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, spike_mode: str='LIF', dvs: bool=False):
        super().__init__()
        self.attn = MS_SSA_Conv(in_channels, num_heads, spike_mode, dvs)
        self.mlp = MS_MLP_Conv(in_channels, 2*in_channels, spike_mode=spike_mode)

    def forward(self, x: Tensor) -> Tensor:
        x_attn, attn = self.attn(x)
        x = self.mlp(x_attn)
        return x, attn

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_weight: bool=True, use_init: bool=False):
        super(GraphConvLayer, self).__init__()
        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, x0: Tensor) -> Tensor:
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)
        return x


class GConv(nn.Module):
    def __init__(
        self, in_channels: int, hid_channels: int, num_layers: int=2, 
        dropout: float=0.5, use_bn: bool=True, use_residual: bool=True,
        use_weight: bool=True, use_init: bool=False, use_act: bool=True
    ):
        super(GConv, self).__init__()
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

        self.lin = nn.Linear(in_channels, hid_channels)

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hid_channels))
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphConvLayer(hid_channels, hid_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hid_channels))

    def reset_parameters(self):
        self.lin.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters()
        
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.lin(x)
        
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_ = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


class SpikeGraphTransformer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hid_channels: int, num_heads: int,
        trans_num_layers: int, drop_rate: float=0.0, T: int=2, spike_mode: str='LIF',
        dvs_mode: bool=False, TET: bool=False, gnn_num_layers: int=2, gnn_dropout: float=0.5, 
        gnn_use_weight: bool=True, gnn_use_init: bool=False, gnn_use_bn: bool=True, gnn_use_residual: bool=True, 
        gnn_use_act: bool=True, graph_weight: float=0.8, aggregate: str='add', **kwargs
    ):
        super().__init__()
        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        self.drop_rate = drop_rate
        self.activation = F.relu

        # graph branch
        if self.graph_weight > 0:
            self.graph_conv = GConv(
                in_channels, out_channels, gnn_num_layers, gnn_dropout, gnn_use_bn,
                gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act
            )
            if aggregate == 'add':
                self.fc = nn.Linear(out_channels, out_channels)
            elif aggregate == 'cat':
                self.fc = nn.Linear(2 * out_channels, out_channels)
            else:
                raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.head = nn.Linear(hid_channels, out_channels) if out_channels > 0 else nn.Identity()
        self.lin = nn.Linear(in_channels, hid_channels)
        self.bn = nn.LayerNorm(hid_channels)
        self.blocks = nn.ModuleList([MS_Block_Conv(hid_channels, num_heads, spike_mode, dvs_mode) for j in range(trans_num_layers)])

        # classification head
        if spike_mode == 'LIF':
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'PLIF':
            self.head_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')
        
        # self.apply(self._init_weights)
        self.params1 = list(self.lin.parameters())
        self.params1.extend(list(self.bn.parameters()))
        self.params1.extend(list(self.blocks.parameters()))
        self.params1.extend(list(self.head_lif.parameters()))
        self.params1.extend(list(self.head.parameters()))
        if self.graph_weight > 0:
            self.params2 = list(self.graph_conv.parameters())
            self.params2.extend(list(self.fc.parameters()))
        else:
            self.params2 = []

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     if isinstance(m, nn.Conv1d):
    #         torch.nn.init.trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm1d):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def reset_parameters(self):
        self.graph_conv.reset_parameters()
        self.fc.reset_parameters()
        
        self.head.reset_parameters()
        self.lin.reset_parameters()
        self.bn.reset_parameters()
        
        for block in self.blocks:
            block.reset_parameters()
    
    def forward_features(self, x):
        x = self.lin(x)
        x = self.bn(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        for block in self.blocks:
            x, _ = block(x)
        return x

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if len(x.shape) < 3:
            x1 = (x.unsqueeze(0)).repeat(int(self.T), 1, 1)
        else:
            x1 = x.transpose(0, 1).contiguous()

        x1 = self.forward_features(x1)
        x1 = self.head_lif(x1)
        x1 = self.head(x1)  # T,B,D
        
        if not self.TET:
            x1 = x1.mean(0)
        
        if self.graph_weight > 0:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
            x = self.fc(x)
        else:
            x = x1
        
        reset_net(self)
        return x