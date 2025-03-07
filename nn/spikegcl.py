from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge


def reset_net(net: nn.Module, reset_type: str = "subtract"):
    for m in net.modules():
        if hasattr(m, "reset"):
            m.reset(reset_type=reset_type)


def heaviside(x: torch.Tensor):
    return x.ge(0)


def gaussian(x, mu, sigma):
    """
    Gaussian PDF with broadcasting.
    """
    return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (
        sigma * torch.sqrt(2 * torch.tensor(pi))
    )


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
        sg = (1.0 - sgax) * sgax * alpha
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


SURROGATE = {
    "sigmoid": sigmoidspike,
    "triangle": trianglespike,
    "arctan": arctanspike,
    "mg": mgspike,
    "super": superspike,
}


class IF(nn.Module):
    def __init__(
        self, v_threshold=1.0, v_reset=0.0, alpha=1.0, surrogate="triangle", detach=True,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach = detach
        self.surrogate = SURROGATE.get(surrogate)
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.v = 0.0
        self.reset()

    def reset(self, reset_type: str = "subtract"):
        assert reset_type in ["zero", "subtract"]
        if reset_type == "zero":
            self.v = 0
        else:
            self.v = self.v - self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v += dv
        # 2. fire
        spike = self.surrogate(self.v, self.v_threshold, self.alpha)
        if self.detach:
            detached_spike = spike.detach()
            v = self.v.detach()
        else:
            v = self.v
            detached_spike = spike
        # 3. reset
        self.v = (1 - detached_spike) * v + detached_spike * self.v_reset
        return spike

class LIF(nn.Module):
    def __init__(
        self,
        tau=1.0,
        v_threshold=1.0,
        v_reset=0.0,
        alpha=1.0,
        surrogate="triangle",
        detach=True,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach = detach
        self.surrogate = SURROGATE.get(surrogate)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.v = 0.0
        self.reset()

    def reset(self, reset_type: str = "subtract"):
        assert reset_type in ["zero", "subtract"]
        if reset_type == "zero":
            self.v = 0
        else:
            self.v = self.v - self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_threshold, self.alpha)
        if self.detach:
            detached_spike = spike.detach()
            v = self.v.detach()
        else:
            v = self.v
            detached_spike = spike
        # 3. reset
        self.v = (1 - detached_spike) * v + detached_spike * self.v_reset
        return spike
        
class PLIF(nn.Module):
    def __init__(
        self,
        tau=1.0,
        v_threshold=1.0,
        v_reset=0.0,
        alpha=1.0,
        surrogate="triangle",
        detach=True,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach = detach
        self.surrogate = SURROGATE.get(surrogate)
        self.register_parameter(
            "tau", nn.Parameter(torch.as_tensor(tau, dtype=torch.float32))
        )
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.v = 0.0

    def reset(self, reset_type: str = "subtract"):
        assert reset_type in ["zero", "subtract"]
        if reset_type == "zero":
            self.v = 0
        else:
            self.v = self.v - self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_threshold, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        if self.detach:
            detached_spike = spike.detach()
            v = self.v.detach()
        else:
            v = self.v
            detached_spike = spike
        # 3. reset
        self.v = (1 - detached_spike) * v + detached_spike * self.v_reset
        return spike


def creat_activation_layer(activation):
    if activation is None:
        return torch.nn.Identity()
    elif activation == "relu":
        return torch.nn.ReLU()
    elif activation == "elu":
        return torch.nn.ELU()
    else:
        raise ValueError("Unknown activation")


def creat_snn_layer(
    alpha=2.0,
    surrogate="sigmoid",
    v_threshold=5e-3,
    snn="PLIF",
):
    tau = 1.0

    if snn == "LIF":
        return LIF(tau, alpha=alpha, surrogate=surrogate, v_threshold=v_threshold, detach=True)
    elif snn == "PLIF":
        return PLIF(tau, alpha=alpha, surrogate=surrogate, v_threshold=v_threshold, detach=True)
    elif snn == "IF":
        return IF(alpha=alpha, surrogate=surrogate, v_threshold=v_threshold, detach=True)
    else:
        raise ValueError("Unknown SNN")


class SpikeGCL(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        T: int = 32,
        alpha: float = 2.0,
        surrogate: str="sigmoid",
        v_threshold: float=5e-3,
        neuron: str="PLIF",
        reset: str="zero",
        act: str="elu",
        dropedge: float=0.2,
        dropout: float=0.5,
        bn: bool=True,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.snn = creat_snn_layer(
            alpha=alpha,
            surrogate=surrogate,
            v_threshold=v_threshold,
            snn=neuron,
        )
        bn = torch.nn.BatchNorm1d if bn else torch.nn.Identity

        in_channels = [
            x.size(0) for x in torch.chunk(torch.ones(in_channels), T)
        ]
        for channel in in_channels:
            self.convs.append(GCNConv(channel, hid_channels))
            self.bns.append(bn(channel))

        self.shared_bn = bn(hid_channels)
        self.shared_conv = GCNConv(hid_channels, hid_channels)

        self.lin = torch.nn.Linear(hid_channels, hid_channels, bias=False)
        self.act = creat_activation_layer(act)
        self.drop_edge = dropedge
        self.T = T
        self.dropout = torch.nn.Dropout(dropout)
        self.reset = reset

    def encode(self, x, edge_index, edge_weight=None):
        chunks = torch.chunk(x, self.T, dim=1)
        xs = []
        for i, x in enumerate(chunks):
            x = self.dropout(x)
            x = self.bns[i](x)
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            x = self.dropout(x)
            x = self.shared_bn(x)
            x = self.shared_conv(x, edge_index, edge_weight)
            x = self.snn(x)
            xs.append(x)
        self.snn.reset(self.reset)
        return xs

    def decode(self, spikes):
        xs = []
        for spike in spikes:
            xs.append(self.lin(spike).sum(1))
        return xs

    def forward(self, x, edge_index, edge_weight=None):
        edge_index2, mask2 = dropout_edge(edge_index, p=self.drop_edge)

        if edge_weight is not None:
            edge_weight2 = edge_weight[mask2]
        else:
            edge_weight2 = None

        x2 = x[:, torch.randperm(x.size(1))]

        s1 = self.encode(x, edge_index, edge_weight)
        s2 = self.encode(x2, edge_index2, edge_weight2)

        z1 = self.decode(s1)
        z2 = self.decode(s2)
        return z1, z2

    def loss(self, postive, negative, margin=0.0):
        loss = F.margin_ranking_loss(
            postive, negative, target=torch.ones_like(postive), margin=margin
        )
        return loss