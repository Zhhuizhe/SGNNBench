import time
import numpy as np
from copy import deepcopy
from logzero import logger
from geoopt.optim import RiemannianAdam

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    add_self_loops,
)

from sgnnbenchmark.mlp_evaluator import evaluate_ssl
from sgnnbenchmark.nn import load_model
from sgnnbenchmark.dataset import load_dataset, load_splits
from sgnnbenchmark.functional import compute_loss, compute_metric
from sgnnbenchmark.utils import seed_everything, count_model_params, Config, HETERO_DATASETS


def train_ssl(data, model, optimizer, margin) -> float:
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    view1, view2 = model(data.x, data.edge_index)
    for v1, v2 in zip(view1, view2):
        loss = model.loss(v1, v2, margin)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss


def train(data, model, optimizer, criterion) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index if data.edge_index is not None else data.adj_t)
    loss = compute_loss(out[data.train_indices], data.y[data.train_indices], criterion)
    
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(data, model, metric):
    model.eval()
    
    pred = model(data.x, data.edge_index if data.edge_index is not None else data.adj_t)
    pred, target = pred.cpu(), data.y.cpu()
    
    accs = []
    for indices in [data.valid_indices, data.test_indices]:
        indices = indices.cpu()
        acc = compute_metric(pred[indices], target[indices], metric)
        accs.append(acc)
    return accs


class Exp:
    def __init__(self, configs: Config):
        self._exp_configs = deepcopy(configs)
        
        self._shared_cfg = self._exp_configs.shared
        self._dataset_cfg = self._exp_configs.dataset
        self._loader_cfg = self._exp_configs.loader
        self._opt_cfg = self._exp_configs.opt
        self._model_cfg = self._exp_configs.model
    
    def _run(self, data: Data, model: nn.Module, run: int):
        # Create the new split based on each random seed
        train_indices, valid_indices, test_indices = load_splits(self._dataset_cfg.name, data, run)
        data.train_indices, data.valid_indices, data.test_indices = train_indices, valid_indices, test_indices
        
        model.reset_parameters()
        model, data = model.to(self._shared_cfg.device), data.to(self._shared_cfg.device)
        model_name = self._model_cfg.model_name
        if model_name.lower() == 'msg':
            optimizer = RiemannianAdam(model.parameters(), lr=self._opt_cfg.lr, weight_decay=self._opt_cfg.wd)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self._opt_cfg.lr, weight_decay=self._opt_cfg.wd)
        
        model_size = count_model_params(model)
        logger.info(f'Train, Val and Test Splits: {len(train_indices)}, {len(valid_indices)}, {len(test_indices)}, Model Size (MB): {model_size:.4f}')
        
        best_val_acc = best_test_acc = patience = 0
        for epoch in range(1, self._shared_cfg.epochs + 1):
            if model_name == 'spikegcl':
                loss = train_ssl(data, model, optimizer, self._opt_cfg.margin)
            else:
                loss = train(data, model, optimizer, self._dataset_cfg.criterion) # Training step

            if epoch == 1 or epoch % self._shared_cfg.eval_step == 0:
                start_time = time.time()
                if model_name == 'spikegcl':
                    model.eval()
                    with torch.no_grad():
                        embd = model.encode(data.x, data.edge_index, data.edge_attr)
                        embd = torch.cat(embd, dim=-1)
                    val_acc, test_acc = evaluate_ssl(embd, data, self._dataset_cfg.criterion, self._dataset_cfg.metric_type)
                else:
                    val_acc, test_acc = test(data, model, self._dataset_cfg.metric_type) # Evaluation step
                end_time = time.time()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience = 0
                else:
                    patience += 1
                
                logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Valid: {val_acc:.4%}, '
                            f'Test: {test_acc:.4%}, Best Test: {best_test_acc:.4%}, Elapsed Time: {end_time-start_time:.4f}')
            
            if patience > self._shared_cfg.patience: # Early stop
                break
        return best_val_acc, best_test_acc
    
    def run(self, hyperparameter_sapce=None):
        self._exp_configs.merge_from_dict(hyperparameter_sapce)
        
        if self._dataset_cfg.name in ['Minesweeper', 'Questions', 'ogbn-proteins']:
            self._dataset_cfg['metric_type'] = 'rocauc'
            self._dataset_cfg['criterion'] = 'binary_cross_entropy'
    
        logger.info(self._exp_configs)
        
        # Load Datasets
        data = load_dataset(**self._dataset_cfg)
        logger.info(
            f'**{self._dataset_cfg.name}** Num of Nodes: {data.num_nodes}, Num of Edges: {data.num_edges}, '
            f'Num of Features: {data.num_features}, Num of Classes: {data.num_classes}'
        )
        
        # Load Models
        model = load_model(data, self._model_cfg)
        logger.info(model)
        
        if self._dataset_cfg.name in HETERO_DATASETS:
            edge_index_tmp = to_undirected(data.edge_index)
            edge_index_tmp, _ = remove_self_loops(edge_index_tmp)
            edge_index_tmp, _ = add_self_loops(edge_index_tmp, num_nodes=data.num_nodes)
            data.edge_index = edge_index_tmp

        ######## Training loop ########
        val_accs, test_accs = [], []
        for run in range(0, self._shared_cfg.runs):
            print('#'*25 + f'{run + 1}/{self._shared_cfg.runs}' + '#'*25)
            seed_everything(self._shared_cfg.seed + run)
            
            if self._loader_cfg.enable_loader:
                best_val_acc, best_test_acc = self._run_batch(data)
            else:
                try:
                    best_val_acc, best_test_acc = self._run(data, model, run)
                except Exception as e:
                    print(e)
                    val_accs, test_accs = [999999], [999999]
                    break
            val_accs.append(best_val_acc)
            test_accs.append(best_test_acc)
        
        logger.info(f'Final ACC: {np.mean(test_accs):.4%}±{100*np.std(test_accs):.4f}')
        return 1 / np.mean(val_accs)
