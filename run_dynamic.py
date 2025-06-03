import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from logzero import logger
from sklearn import metrics
from hyperopt import space_eval, fmin, hp, tpe

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sgnnbenchmark.dataset import load_dataset
from sgnnbenchmark.nn_dyn import load_model
from sgnnbenchmark.utils import seed_everything, load_config, create_log_file, gcn_norm_with_k_order


def train(data, loader, model, optimizer, loss_fn, adj_gcn_norm=None):
    model.train()
    
    # Training phase
    total_loss = 0
    for nodes in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        
        pred = model(data, nodes, device, adj_gcn_norm)
        target = data.y[nodes].to(device)
        
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data.train_nodes)


@torch.no_grad()
def test(data, loader, model, adj_gcn_norm=None):
    model.eval()
    
    # Inference phase
    preds, labels = [], []
    for nodes in loader:
        preds.append(model(data, nodes, device, adj_gcn_norm).cpu())
        labels.append(data.y[nodes].cpu())

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = preds.argmax(1)
    metric_macro = metrics.f1_score(labels, preds, average='macro')
    metric_micro = metrics.f1_score(labels, preds, average='micro')
    return metric_macro, metric_micro


def single_run(data, model, optimizer, loss_fn, adj_gcn_norm=None):
    best_test_metric, best_val_metric = None, None
    
    train_loader = DataLoader(data.train_nodes, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data.val_nodes, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data.test_nodes, batch_size=args.batch_size, shuffle=False)
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train(data, train_loader, model, optimizer, loss_fn, adj_gcn_norm)
        
        if epoch == 1 or epoch % args.eval_step == 0:
            val_metric = test(data, val_loader, model, adj_gcn_norm)
            test_metric = test(data, test_loader, model, adj_gcn_norm)
                
            if best_val_metric is None or val_metric[1] >= best_val_metric[1]:
                best_val_metric = val_metric
                best_test_metric = test_metric
                patience = 0
            else:
                patience += 1
        
            end = time.time()
            logger.info(
                f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Elapse Time(s): {end-start:.4}, '
                f'Macro-f1(Val): {val_metric[0]:.4f}, Micro-f1(Val): {val_metric[1]:.4f}, ' 
                f'Macro-f1(Test): {test_metric[0]:.4f}, Micro-f1(Test): {test_metric[1]:.4f}'
            )
            if patience > args.patience:
                break

    return best_val_metric, best_test_metric


def exp(hyperparameter_sapce=None):
    if hyperparameter_sapce is not None:
        for key in args.__dict__.keys():
            if key in hyperparameter_sapce.keys():
                args.__dict__[key] = hyperparameter_sapce[key]        
    logger.info(args)
    
    # Load the dataset
    data = load_dataset(args.root, args.name)
    logger.info(
        f'##{args.name}## Num of Nodes: {data.num_nodes}, Num of features: {data.num_features} '
        f'Num of Class: {data.num_classes}, Num of Steps: {len(data)}'
    )
        
    # Load the model
    model = load_model(data, args).to(device)
    logger.info(model)
    
    adj_gcn_norm = None
    if args.model_name == 'spikinggcn':
        adj_normalized_dir = f'./adj_normalized/{args.name}_K_{args.K}.pt' # Load the nomalized adjacency matrix from the local file
        if os.path.exists(adj_normalized_dir):
            adj_gcn_norm = torch.load(adj_normalized_dir)
        else:
            adj_gcn_norm = gcn_norm_with_k_order(data, args.name, args.K)
    
    final_metrics = {'val_macro': [], 'val_micro': [], 'test_macro': [], 'test_micro': []}
    for run in range(args.runs):
        logger.info('#'*50 + f'{run+1}/{args.runs}' + '#'*50)
        
        seed_everything(args.seed+run)
        data.split_nodes(args.splits-0.05, 0.05, 1-args.splits) # 5% training samples 

        model.reset_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        try:
            best_val_metrics, best_test_metrics = \
                single_run(data, model, optimizer, loss_fn, adj_gcn_norm)
        except Exception as e:
            print(e)
            return 999999
        
        final_metrics['test_macro'].append(best_test_metrics[0])
        final_metrics['test_micro'].append(best_test_metrics[1])
        final_metrics['val_micro'].append(best_val_metrics[1])
    
    macro_f1_list = final_metrics['test_macro']
    micro_f1_list = final_metrics['test_micro']
    logger.info(
        f'Final Best Macro: {np.mean(macro_f1_list):.4%}±{100 * np.std(macro_f1_list):.4f}, '
        f'Final Best Micro: {np.mean(micro_f1_list):.4%}±{100 * np.std(micro_f1_list):.4f}'
    )
    return 1 / np.mean(final_metrics['val_micro']) # Utilize the validation sets to select optimal parameter combinations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./configs/dynamic/dyn_spikinggcn.yaml")
    args = parser.parse_args()
    args = load_config(args)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    create_log_file(args.log_dir)
    
    exp()