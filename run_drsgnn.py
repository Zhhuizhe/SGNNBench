import time
import os
import argparse
import numpy as np
from logzero import logger
from spikingjelly.clock_driven.encoding import PoissonEncoder
from spikingjelly.clock_driven.functional import reset_net

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from torch_geometric.utils import (
    to_undirected,
    to_scipy_sparse_matrix,
    from_scipy_sparse_matrix,
    to_dense_adj,
    add_self_loops,
    remove_self_loops
)
from graphgallery.functional import normalize_adj

from sgnnbenchmark.nn.drsgnn import LIFSpike
from sgnnbenchmark.dataset import load_dataset, load_splits
from sgnnbenchmark.functional import compute_metric
from sgnnbenchmark.utils import create_log_file, load_config, seed_everything, HETERO_DATASETS


@torch.no_grad
def evaluate(
    dataloader: DataLoader, 
    model: nn.Module, 
    encoder: nn.Module, 
    num_classes: int,
    metric: str,
):
    model.eval()
    
    test_sum = correct_sum = 0
    labels_lst, preds_lst = [], []
    for img, label in dataloader:
        img = img.to(device)
        n_imgs = img.shape[0]
        out_spikes_counter = torch.zeros(n_imgs, num_classes).to(device)
        for t in range(args.T):
            out_spikes_counter +=  model(encoder(img[:,:]).float())
        
        reset_net(model)
        labels_lst.append(label.cpu())
        preds_lst.append(out_spikes_counter.cpu())
        # correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
        # test_sum += label.numel()
    acc = compute_metric(preds_lst, labels_lst, metric)
    return acc


def train(train_loader, model, optimizer):
    total_loss = 0
    for x_batch, label_batch in train_loader:
        optimizer.zero_grad()
        x_batch, label_batch = x_batch.to(device), label_batch.to(device)
        pred = model(x_batch)
    
        loss = F.mse_loss(pred, label_batch.float())
        loss.backward()
        optimizer.step()
        model.reset()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def position_encoding(root: str, name: str, pe_dir: str, nb_pos_enc: int):
    data = load_dataset(root, name)
    
    if name in HETERO_DATASETS:
        edge_index_tmp = to_undirected(data.edge_index)
        edge_index_tmp, _ = remove_self_loops(edge_index_tmp)
        edge_index_tmp, _ = add_self_loops(edge_index_tmp, num_nodes=data.num_nodes)
        data.edge_index = edge_index_tmp
    
    # Read dataset and create the positional encoding file
    pe_embd_dir = os.path.join(pe_dir, f'{name}_{nb_pos_enc}.npy')
    if not os.path.exists(pe_embd_dir):
        raise FileNotFoundError()
        
    print('Precalcluating positional encoding...')
    # Load position encoding from local files
    pe = torch.from_numpy(np.load(pe_embd_dir))
    
    # Implement a simplified SGC
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes)
    adj = normalize_adj(adj)

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    adj_dense = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0]
    mat = adj_dense @ adj_dense @ data.x
    
    data.feat_with_pe = torch.cat([mat, pe], dim=-1)
    print('Done!')
    return data


def exp(data, search_space=None):
    if search_space is not None:
        for key, value in search_space.items():
            if key in args.__dict__.keys():
                args.__dict__[key] = value
    logger.info(args)
    
    feat_with_pe = data.feat_with_pe
    val_accs, test_accs = [], []
    for run in range(0, args.runs):
        seed_everything(args.seed + run)

        train_indices, valid_indices, test_indices = load_splits(args.name, data, run)
        train_loader = DataLoader(TensorDataset(feat_with_pe[train_indices], data.y[train_indices]), args.batch_size, shuffle=True, drop_last=False)
        valid_loader = DataLoader(TensorDataset(feat_with_pe[valid_indices], data.y[valid_indices]), args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(TensorDataset(feat_with_pe[test_indices], data.y[test_indices]), args.batch_size, shuffle=False, drop_last=False)
        
        # model_size = count_model_params(model)
        print(f'Train/Val/Test Splits: {len(train_indices)}, {len(valid_indices)}, {len(test_indices)}\n'
                f'Train/Val/Test Loader: {len(train_loader)}, {len(valid_loader)}, {len(test_loader)}')
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_with_pe.size(1), args.hid_channels),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hid_channels, data.num_classes),
            LIFSpike(data.num_classes)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
        encoder = PoissonEncoder()

        best_val_acc = best_test_acc = patience = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            if epoch == 50:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
            if epoch == 80:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001
            
            for img, label in train_loader:
                img = img.to(device)
                label = label.long().to(device)
                label_one_hot = F.one_hot(label, data.num_classes).float()
                optimizer.zero_grad()

                for t in range(args.T):
                    if t == 0: out_spikes_counter = model(encoder(img[:,:]).float())
                    else: out_spikes_counter += model(encoder(img[:,:]).float())
                out_spikes_counter_frequency = out_spikes_counter / args.T

                loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
                # loss = compute_loss(out_spikes_counter_frequency, label, args.criterion)
                loss.backward()
                optimizer.step()
                reset_net(model)
            scheduler.step()
            
            if epoch == 1 or epoch % args.eval_step == 0:
                start_time = time.time()
                val_acc = evaluate(valid_loader, model, encoder, data.num_classes, args.metric_type)    
                test_acc = evaluate(test_loader, model, encoder, data.num_classes, args.metric_type)    
                end_time = time.time()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience = 0
                else:
                    patience += 1
        
                logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Valid: {val_acc:.4%}, Test: {test_acc:.4%}, '
                            f'Best Test: {best_test_acc:.4%}, Elapsed Time: {end_time-start_time:.4f}')

            if patience > args.patience:
                break
        val_accs.append(best_val_acc)
        test_accs.append(best_test_acc)
    logger.info(f'Final ACC: {np.mean(test_accs):.4%}±{100*np.std(test_accs):.4f}')
    return np.mean(val_accs)

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pe_dir', type=str, default='./pe')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--config_dir', type=str, default='./configs/static/drsgnn.yaml')
    args = parser.parse_args()
    args = load_config(args)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Create experimental logs
    create_log_file(args.log_dir)

    data = position_encoding(args.root, args.name, args.pe_dir, args.nb_pos_enc)
    exp(data)
