from typing import Union, List

import torch
import torch.nn as nn
from torch import Tensor


class SAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, concat: bool=False):
        super().__init__()
        self.concat = concat
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)
        
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        
    def forward(self, x: Union[List, Tensor], neigh_x: Union[List, Tensor]) -> Tensor:
        if not isinstance(x, Tensor):
            x = torch.cat(x, dim=0)

        if not isinstance(neigh_x, Tensor):
            neigh_x = torch.cat([torch.mean(h, dim=1) for h in neigh_x], dim=0)
        else:
            neigh_x = torch.mean(neigh_x, dim=1)

        x = self.lin_l(x)
        neigh_x = self.lin_r(neigh_x)
        out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
        return out
