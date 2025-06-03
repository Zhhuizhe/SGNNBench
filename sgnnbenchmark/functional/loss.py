import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_loss(pred: Tensor, target: Tensor, criterion: str):
    if target.ndim == 2 and target.shape[1] == 1:
        target = target.squeeze()
    
    if criterion == 'cross_entropy':
        pred = F.log_softmax(pred, dim=-1)
        # loss =  F.nll_loss(pred, target)
        loss =  nn.NLLLoss()(pred, target)
        return loss
    elif criterion == 'binary_cross_entropy':
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=target.max() + 1) # N -> N×C
            
        return F.binary_cross_entropy_with_logits(pred, target.to(torch.float))
    else:
        raise NotImplementedError