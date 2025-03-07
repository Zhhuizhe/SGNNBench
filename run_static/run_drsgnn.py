import os
import sys
import argparse
import numpy as np
from logzero import logger

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.append("../")
from nn import DRSGNN
from dataset import load_dataset, load_splits
from utils import create_log_file, load_config, precalculate_pe, count_model_params
from functional import compute_metric


@torch.no_grad
def evaluate(data, dataloader, model, metric):
    model.eval()
    pred_list, target_list = [], []
    for node_indices in dataloader:
        pred = model(data.x[node_indices]).cpu()
        target = data.y.squeeze()[node_indices].cpu()
        pred_list.append(pred)
        target_list.append(target)
    
    acc = compute_metric(torch.cat(pred_list, dim=0), torch.cat(target_list, dim=0), metric)
    return acc


def train(data, train_loader, model, optimizer):
    model.train()
    total_loss = 0
    for node_indices in train_loader:
        optimizer.zero_grad()
        pred = model(data.x[node_indices])
    
        label_one_hot = F.one_hot(data.y.squeeze()[node_indices], data.num_classes).float()
        loss = F.mse_loss(pred, label_one_hot)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def exp():
    data = load_dataset(args.root, args.name)
    
    # Read dataset and create the positional encoding file
    pe_embd_dir = os.path.join(args.pe_dir, f'feat+pe_{args.nb_pos_enc}_{args.name}.pt')
    if os.path.exists(pe_embd_dir):
        data.x = torch.load(pe_embd_dir)
    else:    
        print('Precalcluating positional encoding...')
        feat, pe_embd = precalculate_pe(data, args.nb_pos_enc)
        feat_pe_embd = torch.from_numpy(np.concatenate([feat, pe_embd], axis=1)).float()
        torch.save(feat_pe_embd, pe_embd_dir)
        print('Done!')
        data.x = feat_pe_embd

    model = DRSGNN(data.num_features, data.num_classes, args.T)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    model, data = model.to(device), data.to(device)

    train_indices, valid_indices, test_indices = load_splits(args.name, data)
    train_loader = DataLoader(train_indices, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_indices, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_indices, args.batch_size, shuffle=False)
    
    model_size = count_model_params(model)
    print(f'Model Size (MB): {model_size:.4f}')

    best_val_acc = best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        if epoch == 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        loss = train(data, train_loader, model, optimizer)        
        scheduler.step()
        
        if epoch == 1 or epoch % args.eval_step == 0:
            val_acc = evaluate(data, valid_loader, model, args.metric_type)
            test_acc = evaluate(data, test_loader, model, args.metric_type)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    
            logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                    f'Valid: {val_acc:.4%}, Test: {test_acc:.4%}, Best Test: {best_test_acc:.4%}')

    return best_test_acc

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pe_dir', type=str, default='./pe')
    parser.add_argument('--log_dir', type=str, default='../logs')
    parser.add_argument('--config_dir', type=str, default='../configs_static/drsgnn.yaml')
    args = parser.parse_args()
    args = load_config(args)
    
    create_log_file(args.log_dir)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    exp()
