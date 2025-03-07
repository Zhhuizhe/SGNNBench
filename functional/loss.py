import torch
import torch.nn.functional as F
from torch import Tensor


def compute_loss(pred: Tensor, target: Tensor, criterion: str):
    if target.ndim == 2 and target.shape[1] == 1:
        target = target.squeeze()
    
    if criterion == 'cross_entropy':
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, target)
    elif criterion == 'binary_cross_entropy':
        return F.binary_cross_entropy_with_logits(pred, target.to(torch.float))
    else:
        raise NotImplementedError