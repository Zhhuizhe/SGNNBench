import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from .neuron import reset_net


class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_Conv(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            drop=0.0,
            spike_mode="lif",
            layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        # todo 暂时用一维卷积，后面可以改成线性层
        # self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        # self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_conv =nn.Linear(in_features, self.hidden_features)
        # self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.fc2_conv =nn.Linear(self.hidden_features, out_features)


        self.fc2_bn = nn.BatchNorm1d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        # T, B, C, H, W = x.shape
        T, B, D = x.shape
        identity = x

        x = self.fc1_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        # x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        # x = self.fc1_bn(x).reshape(T, B, self.c_hidden, D).contiguous()
        x = self.fc1_bn(x)
        x=x.reshape(T, B, self.hidden_features).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        # x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        x = self.fc2_bn(x).reshape(T, B, D).contiguous()

        x = x + identity
        return x, hook

    def reset_parameters(self):
        for module in self.modules():
            module.reset_parameters()


class MS_SSA_Conv(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            proj_drop=0.0,
            mode="direct_xor",
            spike_mode="lif",
            dvs=False,
            layer=0,
    ):
        super().__init__()
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        # self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        # self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_conv = nn.Linear(dim, dim * self.num_heads)

        # self.q_conv = nn.Conv1d(1, dim, kernel_size=1, stride=1, bias=False)
        # self.q_conv = nn.Conv1d(dim, kernel_size=1, stride=1, bias=False)
        # self.q_bn = nn.BatchNorm2d(dim)
        self.q_bn = nn.BatchNorm1d(dim * self.num_heads)
        if spike_mode == "lif":
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.q_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        # self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        # self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_conv = nn.Linear(dim, dim * self.num_heads)
        # self.k_bn = nn.BatchNorm2d(dim)
        self.k_bn = nn.BatchNorm1d(dim * self.num_heads)
        if spike_mode == "lif":
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.k_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        # self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        # self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_conv = nn.Linear(dim, dim * self.num_heads)

        # self.v_bn = nn.BatchNorm2d(dim)
        self.v_bn = nn.BatchNorm1d(dim * self.num_heads)
        if spike_mode == "lif":
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.v_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        if spike_mode == "lif":
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.attn_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )

        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )
        # self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        # self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_conv = nn.Linear(dim * self.num_heads, dim)
        # self.proj_bn = nn.BatchNorm2d(dim)
        self.proj_bn = nn.BatchNorm1d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, D = x.shape
        # print(x.shape)
        identity = x
        # N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()
        x_for_qkv = x.flatten(0, 1)  # T*B,D
        q_conv_out = self.q_conv(x_for_qkv)  # T*B,D* self.num_heads
        # q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, D, self.num_heads).permute(0, 1, 3,
                                                                                    2).contiguous()  # T, B,self.num_heads, D
        q = self.q_lif(q_conv_out)

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        k_conv_out = self.k_conv(x_for_qkv)
        # k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, D, self.num_heads).permute(0, 1, 3,
                                                                                    2).contiguous()  # T, B,self.num_heads, D
        k = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        v_conv_out = self.v_conv(x_for_qkv)
        # v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        # v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, D).contiguous()
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, D, self.num_heads).permute(0, 1, 3,
                                                                                    2).contiguous()  # T, B,self.num_heads, D
        v = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = x.reshape(T, B, D * self.num_heads).contiguous()  # T, B,self.num_heads* D
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, D)
            .contiguous()
        )  # T, B,D

        x = x + identity
        return x, v, hook

    def reset_parameters(self):
        for module in self.modules():
            module.reset_parameters()


class MS_Block_Conv(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            norm_layer=nn.LayerNorm,
            attn_mode="direct_xor",
            spike_mode="lif",
            dvs=False,
            layer=0,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            # attn_drop=attn_drop,
            proj_drop=drop,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook

    def reset_parameters(self):
        for module in self.modules():
            module.reset_parameters()


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
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

    def forward(self, x, edge_index, x0):
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
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_init=False, use_act=True):
        super(GConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

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
        self,
        feature_size=128,
        num_classes=11,
        hid_channels=128,
        num_heads=1,
        mlp_ratios=2,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        T=2,
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        gnn_num_layers=2, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
        gnn_use_residual=True, gnn_use_act=True,
        graph_weight=0.8, aggregate='add'

    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        # self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        self.embed_dims = hid_channels
        self.mlp_ratios = mlp_ratios

        # graph branch
        if self.graph_weight > 0:
            self.graph_conv = GConv(feature_size, num_classes, gnn_num_layers, gnn_dropout, gnn_use_bn,
                                    gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
            if aggregate == 'add':
                self.fc = nn.Linear(num_classes, num_classes)
            elif aggregate == 'cat':
                self.fc = nn.Linear(2 * num_classes, num_classes)
            else:
                raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(feature_size, hid_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hid_channels))
        self.activation = F.relu
        self.drop_rate = drop_rate

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=hid_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )

        # setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.head = (
            nn.Linear(hid_channels, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)
        self.params1 = list(self.fcs.parameters())
        self.params1.extend(list(self.bns.parameters()))
        self.params1.extend(list(blocks.parameters()))
        self.params1.extend(list(self.head_lif.parameters()))
        self.params1.extend(list(self.head.parameters()))
        if self.graph_weight > 0:
            self.params2 = list(self.graph_conv.parameters())
            self.params2.extend(list(self.fc.parameters()))
        else:
            self.params2 = []

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv1d):
            # trunc_normal_(m.weight, std=0.02)
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        T, B, D = x.shape
        x = self.fcs[0](x)
        # if self.use_bn:
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        block = getattr(self, f"block")
        for blk in block:
            x, _, hook = blk(x, hook=hook)
        return x, hook

    def forward(self, x, edge_index=None, hook=None):

        if len(x.shape) < 3:
            x1 = (x.unsqueeze(0)).repeat(int(self.T), 1, 1)
        else:
            x1 = x.transpose(0, 1).contiguous()

        x1, hook = self.forward_features(x1, hook=hook)
        x1 = self.head_lif(x1)
        if hook is not None:
            hook["head_lif"] = x1.detach()
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
        # return x, hook
        reset_net(self)
        return x