import sys
import argparse
import numpy as np
from logzero import logger
from geoopt.optim import RiemannianAdam

import torch

from mlp_evaluator import evaluate_ssl
sys.path.append('../')
from nn import load_model
from dataset import load_dataset, load_splits
from functional import compute_loss, compute_metric
from utils import create_log_file, seed_everything, count_model_params, configs


def train_ssl(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    view1, view2 = model(data.x, data.edge_index)
    for v1, v2 in zip(view1, view2):
        loss = model.loss(v1, v2, configs.opt.margin)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss


def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = compute_loss(out[data.train_indices], data.y[data.train_indices], criterion)
    
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(data, model, metric):
    model.eval()
    
    pred = model(data.x, data.edge_index)
    pred, target = pred.cpu(), data.y.cpu()
    
    accs = []
    for indices in [data.valid_indices, data.test_indices]:
        indices = indices.cpu()
        acc = compute_metric(pred[indices], target[indices], metric)
        accs.append(acc)
    return accs


def exp():
    logger.info(configs)
    data = load_dataset(configs.dataset.root, configs.dataset.name)

    ### Training loop ###
    val_accs, test_accs = [], []
    for run in range(0, configs.shared.runs):
        seed_everything(configs.shared.seed + run)
        
        train_indices, valid_indices, test_indices = load_splits(configs.dataset.name, data)
        data.train_indices, data.valid_indices, data.test_indices = train_indices, valid_indices, test_indices
        
        model_name = configs.model_name
        model = load_model(model_name, data.num_features, data.num_classes, configs.model)
        if model_name.lower() == 'spikegt': # Notice: we follow the same optimizer setting with the original implementation of SpikeGT
            optimizer = torch.optim.Adam([
                {'params': model.params1, 'weight_decay': configs.opt.trans_weight_decay},
                {'params': model.params2, 'weight_decay': configs.opt.gnn_weight_decay}
            ], lr=configs.opt.lr)
        elif model_name.lower() == 'msg':
            optimizer = RiemannianAdam(model.parameters(), lr=configs.opt.lr, weight_decay=configs.opt.wd)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=configs.opt.lr, weight_decay=configs.opt.wd)
        
        model_size = count_model_params(model)
        logger.info(f'Num of Nodes: {data.num_nodes}, Num of Edges: {data.num_edges}, '
                    f'Num of Features: {data.num_features}, Model Size (MB): {model_size:.4f}')
        model, data = model.to(device), data.to(device)
        
        best_val_acc = best_test_acc = patience = 0
        for epoch in range(1, configs.shared.epochs + 1):
            if model_name == 'spikegcl':
                loss = train_ssl(data, model, optimizer)
            else:
                loss = train(data, model, optimizer, configs.dataset.criterion) # Training step

            if epoch == 1 or epoch % configs.shared.eval_step == 0:
                if model_name == 'spikegcl':
                    model.eval()
                    with torch.no_grad():
                        embd = model.encode(data.x, data.edge_index, data.edge_attr)
                        embd = torch.cat(embd, dim=-1)
                    val_acc, test_acc = evaluate_ssl(embd, data, configs.dataset.criterion, configs.dataset.metric_type)
                else:
                    val_acc, test_acc = test(data, model, configs.dataset.metric_type) # Evaluation step
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience = 0
                else:
                    patience += 1
                
                logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                    f'Valid: {val_acc:.4%}, Test: {test_acc:.4%}, Best Test: {best_test_acc:.4%}')
            
            if patience > configs.shared.patience: # Early stop
                break
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    logger.info(f"Final ACC: {np.mean(test_accs):.4%}±{100*np.std(test_accs):.4f}")
    return np.mean(val_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--config_dir", type=str, default="../configs_static/msg.yaml")
    args = parser.parse_args()
    configs = configs.load_config_from_file(args.config_dir)
    
    create_log_file(args.log_dir)
    
    device = torch.device(f'cuda:{configs.shared.gpu}' if torch.cuda.is_available() else 'cpu')
    
    exp()
