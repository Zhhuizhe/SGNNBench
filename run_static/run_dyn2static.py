import sys
import argparse
import numpy as np
from logzero import logger

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

sys.path.append('../')
from functional import compute_loss, compute_metric
from dataset import load_dataset, load_splits
from utils import create_log_file, seed_everything, load_config, count_model_params
from nn import SiGNN, SpikeNet


def load_model(model_name: str, data: Data) -> nn.Module:
    model = None
    if model_name.lower() == 'spikenet':
        model = SpikeNet(
            data, args.T, args.hids, args.sizes, dropout=args.dropout, act=args.neuron,
        )
    elif model_name.lower() == 'signn':
        model = SiGNN(
            data, args.T, hids=args.hids, sizes=args.sizes, dropout=args.dropout, 
            act=args.neuron, nchannels=args.nchannels, invth=args.v_threshold
        )
    else:
        raise NotImplementedError
    return model


def train(data, train_loader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    for node_indices in train_loader:
        optimizer.zero_grad()
        node_indices = node_indices.to(device)
        out = model(data.x, node_indices)
        
        loss = compute_loss(out, data.y[node_indices], criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def test(data, model, metric):
    model.eval()
    
    accs = []
    for indices in [data.valid_indices, data.test_indices]:
        pred = model(data.x, indices).cpu()
        
        target = data.y[indices].cpu()
        acc = compute_metric(pred, target, metric)
        accs.append(acc)
    return accs


def exp():
    logger.info(args)
    data = load_dataset(args.root, args.name)

    ### Training loop ###
    val_accs, test_accs = [], []
    for run in range(0, args.runs):
        seed_everything(args.seed + run)
        
        train_indices, valid_indices, test_indices = load_splits(args.name, data)
        data.train_indices, data.valid_indices, data.test_indices = train_indices, valid_indices, test_indices
        train_loader = DataLoader(train_indices, batch_size=args.batch_size, shuffle=True)
        # valid_loader = DataLoader(valid_indices, batch_size=args.batch_size, shuffle=False)
        # test_loader = DataLoader(test_indices, batch_size=args.batch_size, shuffle=False)
        
        model = load_model(args.model_name, data)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
        model_size = count_model_params(model)
        logger.info(f'Num of Nodes: {data.num_nodes}, Num of Edges: {data.num_edges}, '
                    f'Num of Features: {data.num_features}, Model Size (MB): {model_size:.4f}')
        model, data = model.to(device), data.to(device)
    
        best_val_acc = best_test_acc = patience = 0
        for epoch in range(args.epochs):
            loss = train(data, train_loader, model, optimizer, args.criterion)

            if epoch % args.eval_step == 0:
                (val_acc, test_acc) = test(data, model, args.metric_type)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience = 0
                else:
                    patience += 1
                
                logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                    f'Valid: {val_acc:.4%}, Test: {test_acc:.4%}, Best Test: {best_test_acc:.4%}')

            if patience > args.patience:
                break
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    logger.info(f'Final ACC: {np.mean(best_test_acc):.4%}±{100*np.std(best_test_acc):.4f}')
    return np.mean(val_accs)


def hyperparam_search(search_space=None):
    if search_space is not None:
        for key, value in search_space.items():
            if key in args.__dict__.keys():
                args.__dict__[key] = value
    logger.info(args)
    
    output_metrics = exp()
    return 1 / output_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='../logs')
    parser.add_argument('--config_dir', type=str, default='../configs_static/spikenet.yaml')
    args = parser.parse_args()
    args = load_config(args)

    create_log_file(args.log_dir)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    exp()
