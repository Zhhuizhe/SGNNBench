import math
import os.path as osp
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, namedtuple
from typing import Union, Optional, List
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    Actor,
    DeezerEurope,
    WikipediaNetwork,
    HeterophilousGraphDataset
)
from torch_geometric.utils import scatter
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from utils import Config, merge


def ratio_rand_splits(label, num_val=0.2, num_test=0.2):
    num_nodes, device = label.shape[0], label.device
    idx_shuffled = torch.randperm(num_nodes, device=device)
    
    val_split = int(num_nodes * num_val)
    test_split = int(num_nodes * num_test)
    
    train_idx = idx_shuffled[test_split + val_split:]
    val_idx = idx_shuffled[:val_split]
    test_idx = idx_shuffled[val_split:test_split + val_split]
    return [train_idx, val_idx, test_idx]
    

def class_rand_splits(label, label_num_per_class=20, valid_num=500, test_num=1000):
    """use all remaining data points as test data, so test_num will not be used"""
    device = label.device

    idx = torch.arange(label.size(0), device=device)
    train_idx, non_train_idx = [], []
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    val_idx, test_idx = (
        non_train_idx[:valid_num],
        non_train_idx[valid_num : valid_num + test_num],
    )
    return [train_idx, val_idx, test_idx]


def load_splits(name, data, enable_mask=False) -> List:    
    # if name in ['Chameleon', 'Squirrel', 'Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Questions']:
    #     for i in range(data.train_mask.shape[1]):
    #         train_idx = torch.where(data.train_mask[:,i])[0]
    #         val_idx = torch.where(data.val_mask[:,i])[0]
    #         test_idx = torch.where(data.test_mask[:,i])[0]
    #         splits_list.append((train_idx, val_idx, test_idx))    
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        return class_rand_splits(data.y) # train: 20 samples per class, val: 500, test: 1000
    elif name in ['Computers', 'Photo', 'CS', 'Physics', 'Actor', 'Deezer']:
        return ratio_rand_splits(data.y) # 60%/20%/20%
    elif name == "Actor":
        return ratio_rand_splits(data.y, 0.25, 0.25)
    elif name in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
        if enable_mask:
            mask_list = []
            for split in ['train', 'valid', 'test']:
                mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                mask[data.split_idx[split]] = True
                mask_list.append(mask)
            return mask_list
        else:
            return [data.split_idx['train'], data.split_idx['valid'], data.split_idx['test']]
    else:
        raise NotImplementedError


def load_ogbn_dataset(root: Union[Path, str], name: str, return_sparse: bool=False):
    if name in ['ogbn-proteins', 'ogbn-products']:
        dataset = PygNodePropPredDataset(name, root)
    elif name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name, root, T.ToUndirected())
    else:
        raise NotImplementedError(name)
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.split_idx = dataset.get_idx_split()
    
    if name == 'ogbn-proteins':
        # Move edge features to node features
        # Notice: the example code presented ogb is, data.x = data.adj_t.mean(dim=1), the following operation 
        # will bring slightly numerical differences.
        data.x = scatter(data.edge_attr, data.edge_index[0], dim=0, dim_size=data.num_nodes, reduce='mean')
        data.edge_attr = None
        data.num_classes = 112
    if return_sparse:
        data = T.ToSparseTensor()(data)
    return data


def load_pyg_dataset(root: Union[Path, str], name: str):
    if name in ['Cora', 'Pubmed', 'Citeseer']:
        dataset = Planetoid(root, name)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root, name, transform=T.NormalizeFeatures())
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root, name, transform=T.NormalizeFeatures())
    elif name == 'Actor':
        dataset = Actor(root)
    elif name == 'Deezer':
        dataset = DeezerEurope(root)
    elif name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True)
    elif name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']:
        dataset = HeterophilousGraphDataset(root=root, name=name)
    else:
        raise NotImplementedError(name)
    
    data = dataset[0]
    data.num_classes = dataset.num_classes
    return data


# Node-level datasets from PYG and obg
def load_dataset(root: Union[Path, str], name: str, return_sparse: bool=False):
    if 'ogbn' in name:
        data = load_ogbn_dataset(root, name, return_sparse)
    else:
        data = load_pyg_dataset(root, name)
    return data


Data = namedtuple('Data', ['x', 'edge_index'])


def standard_normalization(arr):
    n_steps, n_node, n_dim = arr.shape
    arr_norm = preprocessing.scale(np.reshape(arr, [n_steps, n_node * n_dim]), axis=1)
    arr_norm = np.reshape(arr_norm, [n_steps, n_node, n_dim])
    return arr_norm


def edges_to_adj(edges, num_nodes, undirected=True):
    row, col = edges
    data = np.ones(len(row))
    N = num_nodes
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    if undirected:
        adj = adj.maximum(adj.T)
    adj[adj > 1] = 1
    return adj


class Dataset:
    def __init__(self, name=None, root="~/pirate_data/"):
        self.name = name
        self.root = root
        self.x = None
        self.y = None
        self.num_features = None
        self.adj = []
        self.adj_evolve = []
        self.edges = []
        self.edges_evolve = []

    def _read_feature(self):
        filename = osp.join(self.root, self.name, f"{self.name}.npy")
        if osp.exists(filename):
            return np.load(filename)
        else:
            return None

    def split_nodes(
        self,
        train_size: float = 0.4,
        val_size: float = 0.0,
        test_size: float = 0.6,
        random_state: Optional[int] = None,
    ):
        val_size = 0. if val_size is None else val_size
        assert train_size + val_size + test_size <= 1.0

        y = self.y
        train_nodes, test_nodes = train_test_split(
            torch.arange(y.size(0)),
            train_size=train_size + val_size,
            test_size=test_size,
            random_state=random_state,
            stratify=y)

        if val_size:
            train_nodes, val_nodes = train_test_split(
                train_nodes,
                train_size=train_size / (train_size + val_size),
                random_state=random_state,
                stratify=y[train_nodes])
        else:
            val_nodes = None

        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes

    def split_edges(
        self,
        train_stamp: float = 0.7,
        train_size: float = None,
        val_size: float = 0.1,
        test_size: float = 0.2,
        random_state: int = None,
    ):

        if random_state is not None:
            torch.manual_seed(random_state)

        num_edges = self.edges[-1].size(-1)
        train_stamp = train_stamp if train_stamp >= 1 else math.ceil(len(self) * train_stamp)

        train_edges = torch.LongTensor(np.hstack(self.edges_evolve[:train_stamp]))
        if train_size is not None:
            assert 0 < train_size < 1
            num_train = math.floor(train_size * num_edges)
            perm = torch.randperm(train_edges.size(1))[:num_train]
            train_edges = train_edges[:, perm]

        num_val = math.floor(val_size * num_edges)
        num_test = math.floor(test_size * num_edges)
        testing_edges = torch.LongTensor(np.hstack(self.edges_evolve[train_stamp:]))
        perm = torch.randperm(testing_edges.size(1))

        assert num_val + num_test <= testing_edges.size(1)

        self.train_stamp = train_stamp
        self.train_edges = train_edges
        self.val_edges = testing_edges[:, perm[:num_val]]
        self.test_edges = testing_edges[:, perm[num_val:num_val + num_test]]

    def __getitem__(self, time_index: int):
        x = self.x[time_index]
        edge_index = self.edges[time_index]
        snapshot = Data(x=x, edge_index=edge_index)
        return snapshot

    def __next__(self):
        if self.t < len(self):
            snapshot = self.__getitem__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self

    def __len__(self):
        return len(self.adj)

    def __repr__(self):
        return self.name


class DBLP(Dataset):
    def __init__(self, root='./data', normalize=True):
        super().__init__(name='dblp', root=root)
        edges_evolve, self.num_nodes = self._read_graph()
        x = self._read_feature()
        y = self._read_label()

        if x is not None:
            if normalize:
                x = standard_normalization(x)
            self.num_features = x.shape[-1]
            self.x = torch.FloatTensor(x)

        self.num_classes = y.max() + 1

        edges = [edges_evolve[0]]
        for e_now in edges_evolve[1:]:
            e_last = edges[-1]
            edges.append(np.hstack([e_last, e_now]))

        self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
        self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
        self.edges = [torch.LongTensor(edge) for edge in edges]
        self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately

        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}.txt")
        d = defaultdict(list)
        N = 0
        with open(filename) as f:
            for line in f:
                x, y, t = line.strip().split()
                x, y = int(x), int(y)
                d[t].append((x, y))
                N = max(N, x)
                N = max(N, y)
        N += 1
        edges = []
        for time in sorted(d):
            row, col = zip(*d[time])
            edge_now = np.vstack([row, col])
            edges.append(edge_now)
        return edges, N

    def _read_label(self):
        filename = osp.join(self.root, self.name, "node2label.txt")
        nodes = []
        labels = []
        with open(filename) as f:
            for line in f:
                node, label = line.strip().split()
                nodes.append(int(node))
                labels.append(label)

        nodes = np.array(nodes)
        labels = LabelEncoder().fit_transform(labels)

        assert np.allclose(nodes, np.arange(nodes.size))
        return labels


class Tmall(Dataset):
    def __init__(self, root='./data', normalize=True):
        super().__init__(name='tmall', root=root)
        edges_evolve, self.num_nodes = self._read_graph()
        x = self._read_feature()

        y, labeled_nodes = self._read_label()
        # reindexing
        others = set(range(self.num_nodes)) - set(labeled_nodes.tolist())
        new_index = np.hstack([labeled_nodes, list(others)])
        whole_nodes = np.arange(self.num_nodes)
        mapping_dict = dict(zip(new_index, whole_nodes))
        mapping = np.vectorize(mapping_dict.get)(whole_nodes)
        edges_evolve = [mapping[e] for e in edges_evolve]

        edges_evolve = merge(edges_evolve, step=10)

        if x is not None:
            if normalize:
                x = standard_normalization(x)
            self.num_features = x.shape[-1]
            self.x = torch.FloatTensor(x)

        self.num_classes = y.max() + 1

        edges = [edges_evolve[0]]
        for e_now in edges_evolve[1:]:
            e_last = edges[-1]
            edges.append(np.hstack([e_last, e_now]))

        self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
        self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
        self.edges = [torch.LongTensor(edge) for edge in edges]
        self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately

        self.mapping = mapping
        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}.txt")
        d = defaultdict(list)
        N = 0
        with open(filename) as f:
            for line in tqdm(f, desc='loading edges'):
                x, y, t = line.strip().split()
                x, y = int(x), int(y)
                d[t].append((x, y))
                N = max(N, x)
                N = max(N, y)
        N += 1
        edges = []
        for time in sorted(d):
            row, col = zip(*d[time])
            edge_now = np.vstack([row, col])
            edges.append(edge_now)
        return edges, N

    def _read_label(self):
        filename = osp.join(self.root, self.name, "node2label.txt")
        nodes = []
        labels = []
        with open(filename) as f:
            for line in tqdm(f, desc='loading nodes'):
                node, label = line.strip().split()
                nodes.append(int(node))
                labels.append(label)

        labeled_nodes = np.array(nodes)
        labels = LabelEncoder().fit_transform(labels)
        return labels, labeled_nodes


class Patent(Dataset):
    def __init__(self, root='./data', normalize=True):
        super().__init__(name='patent', root=root)
        edges_evolve = self._read_graph()
        y = self._read_label()
        edges_evolve = merge(edges_evolve, step=2)
        x = self._read_feature()

        if x is not None:
            if normalize:
                x = standard_normalization(x)
            self.num_features = x.shape[-1]
            self.x = torch.FloatTensor(x)

        self.num_nodes = y.size
        self.num_features = x.shape[-1]
        self.num_classes = y.max() + 1

        edges = [edges_evolve[0]]
        for e_now in edges_evolve[1:]:
            e_last = edges[-1]
            edges.append(np.hstack([e_last, e_now]))

        self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
        self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
        self.edges = [torch.LongTensor(edge) for edge in edges]
        self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately

        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}_edges.json")
        time_edges = defaultdict(list)
        with open(filename) as f:
            for line in tqdm(f, desc='loading patent_edges'):
                # src nodeID, dst nodeID, date, src originalID, dst originalID
                src, dst, date, _, _ = eval(line)
                date = date // 1e4
                time_edges[date].append((src, dst))

        edges = []
        for time in sorted(time_edges):
            edges.append(np.transpose(time_edges[time]))
        return edges

    def _read_label(self):
        filename = osp.join(self.root, self.name, f"{self.name}_nodes.json")
        labels = []
        with open(filename) as f:
            for line in tqdm(f, desc='loading patent_nodes'):
                # nodeID, originalID, date, node class
                node, _, date, label = eval(line)
                date = date // 1e4
                labels.append(label - 1)
        labels = np.array(labels)
        return labels
