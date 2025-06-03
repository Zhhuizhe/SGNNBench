import torch
import torch.nn as nn
from torch import Tensor
from spikingjelly.clock_driven.encoding import PoissonEncoder
from spikingjelly.clock_driven.functional import reset_net


#steps = 4
dt = 5
aa = 0.5 # pseudo derivative range
#kappa = 0.2 #cora:0.5
tau = 0.25  # decay factor

class SpikeAct(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu


spikeAct = SpikeAct.apply


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n1, thre):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n1
    v_t1_n1 = u_t1_n1 - thre
    o_t1_n1 = spikeAct(v_t1_n1)
    return u_t1_n1, o_t1_n1


class tdNorm(nn.Module):
    def __init__(self, bn):
        super(tdNorm, self).__init__()
        self.bn = bn

    def forward(self, x):
        x_ = torch.zeros(x.size(), device=x.device)
        x_ = self.bn(x)

        return x_


class LIFSpike(nn.Module):
    def __init__(self, n_labels, kappa=0.2):#cora:kappa=0.5;citeseer:kappa=0.2
        super(LIFSpike, self).__init__()
        init_w = kappa
        self.w = nn.Parameter(torch.rand(n_labels)) #nn.Parameter(torch.tensor(init_w, dtype=torch.float).expand(n_labels))#(torch.tensor(init_w, dtype=torch.float))

    def forward(self, x):
        u = torch.zeros(x.shape , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        u, out = state_update(u, out, x, self.w.tanh())
        return out


def out_state_update(u_t_n1, o_t_n1, W_mul_o_t1_n1, thre, lateral):
    u_t1_n1 = tau * u_t_n1 * (1 - torch.gt(o_t_n1, thre).float()) + W_mul_o_t1_n1 + lateral
    o_t1_n1 = OutspikeAct(u_t1_n1, thre)
    return u_t1_n1, o_t1_n1


class OutSpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thre):
        ctx.save_for_backward(input)
        ctx.threshol = thre
        output = torch.gt(input, thre)
        output = output.float() * input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        thre = ctx.threshol
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < (thre-aa)] = 0
        return grad_input, None

OutspikeAct = OutSpikeAct.apply


def sig_state_update(u_t_n1, o_t_n1, W_mul_o_t1_n1):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n1
    o_t1_n1 = nn.Sigmoid()(u_t1_n1)
    return u_t1_n1, o_t1_n1


# The model are derived from the official repository: https://github.com/hzhao98/DRSGNN 
class DRSGNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, T: int, **kwargs):
        super().__init__()
        self.T = T
        
        self.conv = nn.Sequential(
            PoissonEncoder(),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels, bias=False),
            LIFSpike(out_channels),
        )
        
    def reset_parameters(self):
        for child in self.conv.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
    
    def reset(self):
        reset_net(self.conv)
    
    def forward(self, x: Tensor) -> Tensor:
        out_spikes = []
        for t in range(self.T):
            if t == 0:
                out_spikes = self.conv(x)
            else:
                out_spikes += self.conv(x)
        
        return out_spikes / self.T