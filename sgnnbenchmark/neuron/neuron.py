import math
from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F


gamma = 0.2
thresh_decay = 0.7


def heaviside(x: torch.Tensor):
    return x.ge()


def gaussian(x, mu, sigma):
    """
    Gaussian PDF with broadcasting.
    """
    return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * torch.sqrt(2 * torch.tensor(pi)))


class BaseSpike(torch.autograd.Function):
    """
    Baseline spiking function.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class SuperSpike(BaseSpike):
    """
    Spike function with SuperSpike surrogate gradient from
    "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al. 2018.

    Design choices:
    - Height of 1 ("The Remarkable Robustness of Surrogate Gradient...", Zenke et al. 2021)
    - alpha scaled by 10 ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x.abs()) ** 2
        return grad_input * sg, None


class MultiGaussSpike(BaseSpike):
    """
    Spike function with multi-Gaussian surrogate gradient from
    "Accurate and efficient time-domain classification...", Yin et al. 2021.

    Design choices:
    - Hyperparameters determined through grid search (Yin et al. 2021)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
        sg = (
                1.15 * gaussian(x, zero, alpha)
                - 0.15 * gaussian(x, alpha, 6 * alpha)
                - 0.15 * gaussian(x, -alpha, 6 * alpha)
        )
        return grad_input * sg, None


class TriangleSpike(BaseSpike):
    """
    Spike function with triangular surrogate gradient
    as in Bellec et al. 2020.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = torch.nn.functional.relu(1 - alpha * x.abs())
        return grad_input * sg, None


class ArctanSpike(BaseSpike):
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x * x)
        return grad_input * sg, None


class SigmoidSpike(BaseSpike):

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sgax = (x * alpha).sigmoid_()
        sg = (1. - sgax) * sgax * alpha
        return grad_input * sg, None


def superspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return SuperSpike.apply(x - thresh, alpha)


def mgspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(0.5)):
    return MultiGaussSpike.apply(x - thresh, alpha)


def sigmoidspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return SigmoidSpike.apply(x - thresh, alpha)


def trianglespike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return TriangleSpike.apply(x - thresh, alpha)


def arctanspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return ArctanSpike.apply(x - thresh, alpha)


SURROGATE = {'sigmoid': sigmoidspike, 'triangle': trianglespike, 'arctan': arctanspike,
             'mg': mgspike, 'super': superspike}

    
class BLIF(nn.Module):
    def __init__(self, hid=128, v_threshold=1.0, v_reset=0., alpha=1.0,
                 surrogate='triangle', intensity=1):
        super().__init__()
        self.register_parameter("v_threshold", nn.Parameter(
            torch.as_tensor(v_threshold, dtype=torch.float32)))
        self.v_th = v_threshold
        self.v_th_in = v_threshold
        self.surrogate = SURROGATE.get(surrogate)
        self.v_reset = v_reset
        self.tau = torch.nn.Parameter(torch.full((hid,), 1.0))
        self.dec = torch.nn.Parameter(torch.full((hid,), 1.0))
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        self.act = nn.Sigmoid()
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold.item()
        
    def forward(self, dv):
        tu = self.act(self.tau.unsqueeze(0).expand(dv.shape[0], -1))
        dec = self.act(self.dec.unsqueeze(0).expand(dv.shape[0], -1))
        # 1. charge
        self.v = self.v - (self.v - self.v_reset) * tu + (1 - tu) * dv
        
        # 2. bidirectional fire
        spike = self.surrogate(self.v, self.v_th, self.alpha) + self.surrogate(-self.v_th, self.v, self.alpha)  
        
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        
        # 4. threhold updates
        self.v_th = self.v_th * dec +  (1 - dec) * spike

        return spike


class IF(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = SURROGATE.get(surrogate)
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v += dv
        # 2. fire
        spike = self.surrogate(self.v, self.v_threshold, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        # 4. threhold updates
        # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
        self.v_th = gamma * spike + self.v_th * thresh_decay
        return spike


class LIF(nn.Module):
    def __init__(self, tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = SURROGATE.get(surrogate)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_th, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        # 4. threhold updates
        # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
        self.v_th = gamma * spike + self.v_th * thresh_decay
        return spike


class PLIF(nn.Module):
    def __init__(self, tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = SURROGATE.get(surrogate)
        self.register_parameter("tau", nn.Parameter(
            torch.as_tensor(tau, dtype=torch.float32)))
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_th, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        # 4. threhold updates
        # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
        self.v_th = gamma * spike + self.v_th * thresh_decay
        return spike


class STLIF(nn.Module):
    def __init__(self, decay=1.0, v_threshold=1.0, alpha=1.0, surrogate='triangle'):
        super().__init__()
        self.v_threshold = v_threshold
        self.surrogate = SURROGATE.get(surrogate)
        
        self.register_buffer("decay", torch.as_tensor(decay, dtype=torch.float32))
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.decay * self.v + dv
        # 2. fire
        spike = self.surrogate(self.v, self.v_th, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v

        return spike


torch.pi = torch.acos(torch.zeros(1)).item() * 2
steps = 4
a = 0.25
Vth = 0.5  #  V_threshold
aa = Vth
tau = 0.25  # exponential decay coefficient
conduct = 0.5 # time-dependent synaptic weight
linear_decay = Vth/(steps * 2)  #linear decay coefficient

gamma_SG = 1.
class SpikeAct_extended(torch.autograd.Function):
    '''
    solving the non-differentiable term of the Heavisde function
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()

        # hu is an approximate func of df/du in linear formulation
        hu = abs(input) < 0.5
        hu = hu.float()

        # arctan surrogate function
        # hu =  1 / ((input * torch.pi) ** 2 + 1)

        # triangles
        # hu = (1 / gamma_SG) * (1 / gamma_SG) * ((gamma_SG - input.abs()).clamp(min=0))

        return grad_input * hu

class ArchAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0.5)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input
    

class LIFSpike_CW(nn.Module):
    '''
    gated spiking neuron
    '''
    def __init__(self, inplace, **kwargs):
        super(LIFSpike_CW, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']
        self.plane = inplace
        #c
        self.alpha, self.beta, self.gamma = [nn.Parameter(- math.log(1 / ((i - 0.5)*0.5+0.5) - 1) * torch.ones(self.plane, dtype=torch.float))
                                                 for i in kwargs['gate']]

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                              for i in kwargs['param'][:-1]]
        self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.plane, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, self.plane), dtype=torch.float))
                                   for i in kwargs['param'][3:]][0]

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        # print(W_mul_o_t_n1.shape, self.alpha[None, :, None, None].sigmoid().shape)
        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().sigmoid(), self.beta.view(1, -1, 1, 1).clone().detach().sigmoid(), self.gamma.view(1, -1, 1, 1).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
        else:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.beta.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1, 1, 1).sigmoid())

        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
        u_t1_n1 = ((1 - al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :, None, None]) + \
                  I_t1 - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))