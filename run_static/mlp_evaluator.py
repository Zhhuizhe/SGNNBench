import sys
import torch
import numpy as np
import torch.nn as nn

sys.path.append('../')
from functional import compute_loss, compute_metric


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

    def reset_parameters(self):
        self.linear.reset_parameters()


def evaluate_ssl(
    embeds, data, cls_criterion: str, cls_metric: str,
    cls_lr: float=0.01, cls_wd: float=5e-4, cls_epochs: int=100, cls_runs: int=1,    
):
    device = embeds.device
    if not isinstance(embeds, torch.Tensor):
        embeds = torch.cat(embeds, dim=1)
    
    all_val_acc, all_test_acc = [], []

    # input_emb = F.normalize(input_emb, dim=1)    # l2 normalize
    gnn_emb_dim = embeds.size(1)

    classifier = Classifier(gnn_emb_dim, data.num_classes).to(device)

    for _ in range(cls_runs):
        best_val_acc, best_test_acc = train_cls(
            classifier, embeds, data.y,
            data.train_indices, data.valid_indices, data.test_indices,
            cls_criterion, cls_metric, lr=cls_lr, weight_decay=cls_wd, epochs=cls_epochs)

        all_val_acc.append(best_val_acc)
        all_test_acc.append(best_test_acc)
    return np.mean(all_val_acc), np.mean(all_test_acc)


@torch.no_grad()
def eval_acc(model, x, y, metric):
    model.eval()
    output = model(x)

    return compute_metric(output, y, metric)


def train_cls(
    cls, x, y, train_mask, val_mask, test_mask,
    cls_criterion, cls_metric, lr=1e-2, weight_decay=1e-5, epochs=100
):
    cls.reset_parameters()
    optimizer = torch.optim.AdamW(
        cls.parameters(), lr=lr, weight_decay=weight_decay)

    train_x, train_y = x[train_mask], y[train_mask]
    val_x, val_y = x[val_mask], y[val_mask]
    test_x, test_y = x[test_mask], y[test_mask]

    best_val_acc, best_test_acc = 0.0, 0.0
    for _ in range(epochs):
        cls.train()
        optimizer.zero_grad()

        output = cls(train_x)
        loss = compute_loss(output, train_y, cls_criterion)
        loss.backward()
        optimizer.step()

        val_acc = eval_acc(cls, val_x, val_y, cls_metric)
        test_acc = eval_acc(cls, test_x, test_y, cls_metric)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    return best_val_acc, best_test_acc
