import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode, LIFNode

from sgnnbenchmark.neuron import reset_net
from sgnnbenchmark.utils import Sampler, add_selfloops
from .conv import SAGEConv


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
        self.attn.reset_parameters()
        self.mlp.reset_parameters()


class SpikeGT(nn.Module):
    def __init__(
        self, data, in_features=128, out_features=11, hid_channels=128, 
        hids=[128], sizes=[5], num_heads=1, p=1.0, dropout=0.0, 
        graph_weight=0.8, depths=2, spike_mode="LIF", dvs_mode=False, concat=False,
    ):
        assert hids[-1] == hid_channels
        super().__init__()
        self.out_features = out_features
        self.depths = depths
        self.dvs = dvs_mode
        self.graph_weight = graph_weight
        self.sizes = sizes
        self.p = p
        
        # Transformer Block
        self.fc = nn.Linear(in_features, hid_channels)
        self.bn = nn.LayerNorm(hid_channels)
        self.dropout = dropout

        self.blocks = nn.ModuleList()
        for _ in range(depths):
            self.blocks.append(MS_Block_Conv(hid_channels, num_heads, spike_mode, dvs_mode))
        
        # MPNN Block
        self.sampler = [Sampler(add_selfloops(adj_matrix))
                        for adj_matrix in data.adj]
        self.sampler_t = [Sampler(add_selfloops(adj_matrix))
                        for adj_matrix in data.adj_evolve]
        
        channels = [in_features] + list(hids)
        self.aggregators = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.aggregators.append(SAGEConv(channels[i], channels[i+1], concat=concat))

        self.snn = LIFNode(tau=2.0)
        self.pooling = nn.Linear(len(data) * hid_channels, out_features)
        
        self.reset_parameters()

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()
        
        for aggregators in self.aggregators:
            aggregators.reset_parameters()
        
        for block in self.blocks:
            block.reset_parameters()

    def forward_features(self, x):
        x = self.fc(x)
        x = self.bn(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for blk in self.blocks:
            x, _ = blk(x)
        return x

    def encode(self, data, nodes, device):
        spikes = []
        sizes = self.sizes
        for time_step in range(len(data)):
            snapshot = data[time_step]
            sampler = self.sampler[time_step]
            sampler_t = self.sampler_t[time_step]

            x = snapshot.x
            embd = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            
            # Message Passing Block
            h = embd.copy()
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
                
                embd_mpnn = aggregator(self_x, neigh_x)
                if i != len(sizes) - 1:
                    embd_mpnn = F.dropout(embd_mpnn, self.dropout, training=self.training)
                    h = torch.split(embd_mpnn, num_nodes[:-(i + 1)])

            embd_trans = embd[0].unsqueeze(0)
            embd_trans = self.forward_features(embd_trans)
            embd_trans = embd_trans[0]
            
            out = self.graph_weight * embd_mpnn + (1 - self.graph_weight) * embd_trans
            out = self.snn(out)
            spikes.append(out)
        spikes = torch.cat(spikes, dim=1)
        reset_net(self)
        return spikes
    
    def forward(self, data, nodes, device, adj):
        spikes = self.encode(data, nodes, device)
        return self.pooling(spikes)