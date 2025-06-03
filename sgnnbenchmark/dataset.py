import os
import math
from collections import defaultdict, namedtuple
from typing import Union, Optional
from pathlib import Path
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    Actor,
    DeezerEurope,
    HeterophilousGraphDataset
)
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.utils import scatter

from sgnnbenchmark.utils import HETERO_DATASETS, merge, standard_normalization, edges_to_adj


def ratio_rand_splits(label, num_val=0.2, num_test=0.2):
    num_nodes, device = label.shape[0], label.device
    idx_shuffled = torch.randperm(num_nodes, device=device)
    
    val_split = int(num_nodes * num_val)
    test_split = int(num_nodes * num_test)
    
    train_idx = idx_shuffled[test_split + val_split:]
    val_idx = idx_shuffled[:val_split]
    test_idx = idx_shuffled[val_split:test_split + val_split]
    return (train_idx, val_idx, test_idx)
    

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
    return (train_idx, val_idx, test_idx)


def load_splits(name: str, data: Data, run: int=None, return_mask: bool=False): 
    if name in ['Chameleon', 'Squirrel']:
        if run > data.train_mask.size(0):
            raise ValueError(f'The number of experiment iterations is larger than the default splits ({run}>{data.train_mask.size(0)})')
        
        device = data.train_mask.device
        node_idx = torch.arange(data.train_mask.size(1), device=device)
        train_idx = node_idx[data.train_mask[run]]
        val_idx = node_idx[data.val_mask[run]]
        test_idx = node_idx[data.test_mask[run]]
    elif name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Questions']:
        if run > data.train_mask.size(1):
            raise ValueError(f'The number of experiment iterations is larger than the default splits ({run}>{data.train_mask.size(0)})')
        
        train_idx = torch.where(data.train_mask[:, run])[0]
        val_idx = torch.where(data.val_mask[:, run])[0]
        test_idx = torch.where(data.test_mask[:, run])[0]
    elif name in ['Cora', 'Citeseer', 'Pubmed']:
        train_idx, val_idx, test_idx = class_rand_splits(data.y) # train: 20 samples per class, val: 500, test: 1000
    elif name in ['Computers', 'Photo', 'CS', 'Physics', 'Actor', 'Deezer']:
        train_idx, val_idx, test_idx = ratio_rand_splits(data.y) # 60%/20%/20%
    elif name == "Actor":
        train_idx, val_idx, test_idx = ratio_rand_splits(data.y, 0.25, 0.25)
    elif name in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
        train_idx, val_idx, test_idx =  (data.split_idx['train'], data.split_idx['valid'], data.split_idx['test'])
    else:
        raise NotImplementedError
    
    if return_mask:
        mask_list = []
        for indices in [train_idx, val_idx, test_idx]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[indices] = True
            mask_list.append(mask)
        return mask_list
    else:
        return [train_idx, val_idx, test_idx]


def load_ogbn_dataset(name: str, root: Union[Path, str]='./data', return_sparse: bool=False):
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


def load_pyg_dataset(
    name: str, 
    root: Union[Path, str]='./data', 
    enable_feat_norm: bool=False, 
    return_sparse: bool=False
):
    transform = T.NormalizeFeatures() if enable_feat_norm else None
    
    if name in ['Cora', 'Pubmed', 'Citeseer']:
        dataset = Planetoid(root, name, transform=transform)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root, name, transform=transform)
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root, name, transform=transform)
    elif name == 'Actor':
        dataset = Actor(root)
    elif name == 'Deezer':
        dataset = DeezerEurope(root)
    elif name in ['Chameleon', 'Squirrel']:
        path = os.path.join(root, f'{name.lower()}_filtered.npz')
        data = np.load(path)
        
        node_feat = torch.as_tensor(data['node_features'])
        labels = torch.as_tensor(data['node_labels'])
        edge_index = torch.as_tensor(data['edges'].T) # E×2 -> 2×E
        
        train_mask = torch.as_tensor(data["train_masks"])
        val_mask = torch.as_tensor(data["val_masks"])
        test_mask = torch.as_tensor(data["test_masks"])

        splits_kwargs = {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}
        data = Data(x=node_feat, edge_index=edge_index, y=labels, **splits_kwargs)
        data.num_classes = labels.max() + 1
        return data
    elif name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']:
        dataset = HeterophilousGraphDataset(root, name)
    else:
        raise NotImplementedError(name)
    
    data = dataset[0]
    data.num_classes = dataset.num_classes
    return data


DynData = namedtuple('Data', ['x', 'edge_index'])


class Dataset:
    def __init__(self, name: str, root: str):
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
        filename = os.path.join(self.root, self.name, f"{self.name}.npy")
        if os.path.exists(filename):
            return np.load(filename)
        else:
            return None

    def split_nodes(
        self,
        train_size: float = 0.4,
        val_size: float = 0.0,
        test_size: float = 0.6,
        seed: Optional[int] = None,
    ):
        val_size = 0. if val_size is None else val_size
        assert train_size + val_size + test_size <= 1.0

        y = self.y
        train_nodes, test_nodes = train_test_split(
            torch.arange(y.size(0)),
            train_size=train_size + val_size,
            test_size=test_size,
            random_state=seed,
            stratify=y)

        if val_size:
            train_nodes, val_nodes = train_test_split(
                train_nodes,
                train_size=train_size / (train_size + val_size),
                random_state=seed,
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
        seed: int = None,
    ):

        if seed is not None:
            torch.manual_seed(seed)

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
        snapshot = DynData(x=x, edge_index=edge_index)
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
    def __init__(self, root: str='./data', normalize: bool=True, step: int=None):
        super().__init__(name='dblp', root=root)
        
        processed_file_dir = os.path.join(self.processed_dir, 'dblp.pt' if step is None else f'dblp_step_{step}.pt')
        if os.path.exists(processed_file_dir):
            processed_file = torch.load(processed_file_dir)
            self.num_features = processed_file['num_features']
            self.num_classes = processed_file['num_classes']
            self.num_nodes = processed_file['num_nodes']
            self.x = processed_file['x']
            self.y = processed_file['y']
            self.adj = processed_file['adj']
            self.adj_evolve = processed_file['adj_evolve']
            self.edges = processed_file['edges']
            self.edges_evolve = processed_file['edges_evolve']
            return 
        
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        
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
        
        torch.save(
            obj={'num_features': self.num_features, 'num_classes': self.num_classes, 'num_nodes': self.num_nodes,
                 'x': self.x, 'y': self.y, 'adj': self.adj, 'adj_evolve': self.adj_evolve, 'edges': self.edges, 'edges_evolve': self.edges_evolve},
            f=processed_file_dir
        )
        
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'dblp', 'processed')
        
    def _read_graph(self):
        filename = os.path.join(self.root, self.name, f"{self.name}.txt")
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
        filename = os.path.join(self.root, self.name, "node2label.txt")
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
    def __init__(self, root: str='./data', normalize: bool=True, step: int=10):
        super().__init__(name='tmall', root=root)
        
        processed_file_dir = os.path.join(self.processed_dir, 'tmall.pt' if step is None else f'tmall_step_{step}.pt')
        if os.path.exists(processed_file_dir):
            processed_file = torch.load(processed_file_dir)
            self.num_features = processed_file['num_features']
            self.num_classes = processed_file['num_classes']
            self.num_nodes = processed_file['num_nodes']
            self.x = processed_file['x']
            self.y = processed_file['y']
            self.adj = processed_file['adj']
            self.adj_evolve = processed_file['adj_evolve']
            self.edges = processed_file['edges']
            self.edges_evolve = processed_file['edges_evolve']
            self.mapping = processed_file['mapping']
            return 
        
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)

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

        edges_evolve = merge(edges_evolve, step=step)

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
        
        torch.save(
            obj={'num_features': self.num_features, 'num_classes': self.num_classes, 'num_nodes': self.num_nodes, 
                 'x': self.x, 'y': self.y, 'mapping': self.mapping, 'adj': self.adj, 'adj_evolve': self.adj_evolve, 
                 'edges': self.edges, 'edges_evolve': self.edges_evolve},
            f=processed_file_dir
        )
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'tmall', 'processed')

    def _read_graph(self):
        filename = os.path.join(self.root, self.name, f"{self.name}.txt")
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
        filename = os.path.join(self.root, self.name, "node2label.txt")
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
    def __init__(self, root: str='./data', normalize: bool=True, step: int=2):
        super().__init__(name='patent', root=root)
        
        processed_file_dir = os.path.join(self.processed_dir, 'patent.pt' if step is None else f'patent_step_{step}.pt')
        if os.path.exists(processed_file_dir):
            processed_file = torch.load(processed_file_dir)
            self.num_features = processed_file['num_features']
            self.num_classes = processed_file['num_classes']
            self.num_nodes = processed_file['num_nodes']
            self.x = processed_file['x']
            self.y = processed_file['y']
            self.adj = processed_file['adj']
            self.adj_evolve = processed_file['adj_evolve']
            self.edges = processed_file['edges']
            self.edges_evolve = processed_file['edges_evolve']
            return 
        
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        
        edges_evolve = self._read_graph()
        edges_evolve = merge(edges_evolve, step=step)
        
        y = self._read_label()
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
        
        torch.save(
            obj={'num_features': self.num_features, 'num_classes': self.num_classes, 'num_nodes': self.num_nodes,
                 'x': self.x, 'y': self.y, 'adj': self.adj, 'adj_evolve': self.adj_evolve, 'edges': self.edges, 'edges_evolve': self.edges_evolve},
            f=processed_file_dir
        )
        
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'patent', 'processed')

    def _read_graph(self):
        filename = os.path.join(self.root, self.name, f"{self.name}_edges.json")
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
        filename = os.path.join(self.root, self.name, f"{self.name}_nodes.json")
        labels = []
        with open(filename) as f:
            for line in tqdm(f, desc='loading patent_nodes'):
                # nodeID, originalID, date, node class
                node, _, date, label = eval(line)
                date = date // 1e4
                labels.append(label - 1)
        labels = np.array(labels)
        return labels


def load_dataset(
    root: Union[Path, str], 
    name: str,
    step: int = None,
    return_sparse: bool=False,
    enable_feat_norm: bool=False,
    **kwargs
) -> Union[Data, Dataset]:
    if name in ['dblp', 'tmall', 'patent']:
        if name == 'dblp':
            data = DBLP(root, step=step)
        elif name == 'tmall':
            data = Tmall(root, step=step)
        elif name == 'patent':
            data = Patent(root, step=step)
    elif 'ogbn' in name:
        data = load_ogbn_dataset(name, root, return_sparse)
    else:
        data = load_pyg_dataset(name, root, enable_feat_norm)
    return data
