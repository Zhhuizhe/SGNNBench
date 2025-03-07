import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)

import torch
import torch.nn.functional as F
from torch import Tensor


def cal_auc(y_pred, y_true):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


def compute_metric(pred: Tensor, target: Tensor, metric: str):
    if target.ndim == 2 and target.shape[1] == 1:
        target = target.squeeze()
    
    if isinstance(pred, list):
        pred = torch.cat(pred)
    
    if isinstance(target, list):
        target = torch.cat(target)
    
    if metric == 'accuracy':
        pred = pred.log_softmax(dim=-1).argmax(dim=-1)
        return (pred == target).sum().item() / len(pred)
    elif metric == 'rocauc':
        if not target.shape == pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')
        
        return cal_auc(pred.detach().cpu(), target.detach().cpu())
    else:
        raise NotImplementedError
        
