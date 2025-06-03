import torch
import torch.nn as nn
from torch import Tensor

from .neuron import (
    IF, 
    LIF, 
    PLIF, 
    BLIF,
    STLIF,
) 
from .riemannian_neuron import Neuron, RiemannianNeuron


__all__ = [
    'IF',
    'LIF',
    'PLIF',
    'BLIF',
    'STLIF',
    'Neuron',
    'RiemannianNeuron',
]


def rate(input: Tensor, num_steps: int) -> Tensor:
    N, D = input.shape
    time_data = input.repeat((num_steps, 1, 1))
    clipped_data = torch.clamp(time_data, min=0, max=1)
    return torch.bernoulli(clipped_data).reshape(N * num_steps, D).to_sparse()


def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()