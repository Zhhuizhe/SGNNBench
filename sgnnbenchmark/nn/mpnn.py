import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, 
    GATConv, 
    SAGEConv, 
    SGConv
)


class MPNN(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, hid_channels, num_layers=3, 
        dropout=0.5, heads=1, K=4, pre_ln=False, enable_pre_lin=False, 
        enable_res=False, ln=False, enable_norm=False, jk=False, conv_type='gcn',
        **kwargs
    ):
        super(MPNN, self).__init__()

        self.dropout = dropout
        self.pre_ln = pre_ln

        self.ln = ln
        self.jk = jk
        self.res = enable_res
        self.pre_linear = enable_pre_lin
        self.bn = enable_norm
        
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        self.lin_in = torch.nn.Linear(in_channels, hid_channels)
        
        if not self.pre_linear:
            if conv_type == 'gat':
                self.local_convs.append(GATConv(in_channels, hid_channels, heads=heads,
                    concat=False, add_self_loops=False, bias=False))
            elif conv_type == 'sage':
                self.local_convs.append(SAGEConv(in_channels, hid_channels))
            elif conv_type == 'sgc':
                self.local_convs.append(SGConv(in_channels, hid_channels, K, add_self_loops=False))
            else:
                self.local_convs.append(GCNConv(in_channels, hid_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hid_channels))
            self.lns.append(torch.nn.LayerNorm(hid_channels))
            self.bns.append(torch.nn.BatchNorm1d(hid_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            num_layers = num_layers - 1
            
        for _ in range(num_layers):
            if conv_type=='gat':
                self.local_convs.append(GATConv(hid_channels, hid_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            elif conv_type=='sage':
                self.local_convs.append(SAGEConv(hid_channels, hid_channels))
            elif conv_type == 'sgc':
                self.local_convs.append(SGConv(hid_channels, hid_channels, K, add_self_loops=False))
            else:
                self.local_convs.append(GCNConv(hid_channels, hid_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hid_channels, hid_channels))
            self.lns.append(torch.nn.LayerNorm(hid_channels))
            self.bns.append(torch.nn.BatchNorm1d(hid_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hid_channels))
                
        self.pred_local = torch.nn.Linear(hid_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()


    def forward(self, x, edge_index):
        
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_final = 0
        
        for i, local_conv in enumerate(self.local_convs):
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x

        x = self.pred_local(x_final)

        return x