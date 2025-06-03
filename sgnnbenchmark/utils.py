import yaml
import random
import logzero
import scipy.sparse as sp
import os.path as osp
import numpy as np
from typing import Union, List
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from texttable import Texttable
from copy import deepcopy
from sklearn import preprocessing

import torch
import torch_cluster
from torch.sparse import mm
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import (
    to_undirected,
    to_torch_csr_tensor,
    get_self_loop_attr,
    get_laplacian,
    to_edge_index,
    scatter, 
    to_torch_sparse_tensor,
)

from sample_neighber import sample_neighber_cpu


HOMO_DATASETS = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CS', 'Physics']
HETERO_DATASETS = ['Actor', 'Squirrel', 'Chameleon', 'Amazon-ratings', 'Roman-empire', 'Minesweeper', 'Questions']
LARGE_DATASETS = ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_model_params(model):
    """
    Return the size of model in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


def wrap_function(objective, args):
    if objective is None:
        return None
    
    def function_wrapper(wrapper_args):
        return objective(args, wrapper_args)
    
    return function_wrapper

    
def tab_printer(args):
    """Function to print the logs in a nice tabular format.
    
    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.
    
    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
    print(t.draw())


def create_log_file(log_dir, filename=None):
    if filename is None:
        filename = datetime.strftime(datetime.now(), "%Y%m%d") + ".log"
    logzero.logfile(osp.join(log_dir, filename))


def load_config(args):
    if not hasattr(args, "config_dir") or not osp.exists(args.config_dir):
        raise FileNotFoundError("Can't find the configuration file.")

    path = Path(args.config_dir)
    config = yaml.safe_load(open(path, "r"))
    for key, val in config.items():
        if hasattr(config[key], "keys"):
            for key, val in config[key].items():
                args.__dict__[key] = val
    return args


class Config(dict):
    def __init__(self, init_dict=None):
        init_dict = {} if init_dict is None else init_dict
        init_dict = self._create_config_tree_from_dict(init_dict)
        super(Config, self).__init__(init_dict)
    
    @classmethod
    def load_config_from_file(cls, args_dir: Union[str, Path]):
        assert osp.exists(args_dir), "Error: The configuration path is not found!"
        
        _, file_extention = osp.splitext(args_dir)
        if file_extention in [".yml", ".yaml"]:
            with open(args_dir, "r") as cfg_file:
                cfg_as_dict = yaml.safe_load(cfg_file)
                return cls(cfg_as_dict)
        else:
            raise NotImplementedError
    
    @classmethod
    def _create_config_tree_from_dict(cls, init_dict: dict):
        dic = deepcopy(init_dict)
        
        for (key, value) in dic.items():
            if isinstance(value, dict):
                dic[key] = cls(value)
        return dic
    
    def merge_from_dict(self, init_dict: dict):
        if init_dict is None:
            return -1
        
        def retrieve_items(key, value, target_dict):
            for key_t in target_dict.keys():
                if key_t == key:
                    target_dict[key_t] = value
                    break
                
                if isinstance(target_dict[key_t], dict):
                    retrieve_items(key, value, target_dict[key_t])
    
        dic = deepcopy(init_dict)
        for key, value in dic.items():
            retrieve_items(key, value, self)
        return 1
            
    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)
    
    # def pop(self, name: str):
    #     if name in self:
    #         return self.pop(name)
    #     else:
    #         raise AttributeError(name)


configs = Config()


class Sampler:
    def __init__(self, adj_matrix: sp.csr_matrix):
        self.rowptr = torch.LongTensor(adj_matrix.indptr)
        self.col = torch.LongTensor(adj_matrix.indices)

    def __call__(self, nodes, size, replace=True):
        nodes = nodes.to('cpu')
        nbr = sample_neighber_cpu(self.rowptr, self.col, nodes, size, replace)
        return nbr
    
    
class RandomWalkSampler:
    def __init__(self, adj_matrix: sp.csr_matrix, p: float = 1.0, q: float = 1.0):
        self.rowptr = torch.LongTensor(adj_matrix.indptr)
        self.col = torch.LongTensor(adj_matrix.indices)
        self.p = p
        self.q = q
        assert torch_cluster, "Please install 'torch_cluster' first."

    def __call__(self, nodes, size, replace=True):
        nbr = torch.ops.torch_cluster.random_walk(self.rowptr, self.col, nodes, size, self.p, self.q)[0][:, 1:] 
        return nbr


def normalize_adj(adj_matrix, rate=-0.5, add_self_loop=True, symmetric=True):
    """Normalize adjacency matrix.

    >>> normalize_adj(adj, rate=-0.5) # return a normalized adjacency matrix

    # return a list of normalized adjacency matrices
    >>> normalize_adj(adj, rate=[-0.5, 1.0]) 

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array or a list of them 
        Single or a list of Scipy sparse matrices or Numpy arrays.
    rate: Single or a list of float scale, optional.
        the normalize rate for `adj_matrix`.
    add_self_loop: bool, optional.
        whether to add self loops for the adjacency matrix.
    symmetric: bool, optional
        whether to use symmetrical  normalization

    Returns
    ----------
    Single or a list of Scipy sparse matrix or Numpy matrices.

    See also
    ----------
    graphgallery.functional.NormalizeAdj          

    """
    def _normalize_adj(adj, r):

        # here a new copy of adj is created
        if add_self_loop:
            adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format='csr')
        else:
            adj = adj.copy()

        if r is None:
            return adj

        degree = np.ravel(adj.sum(1))
        degree[degree < 0] = 0

        degree_power = np.power(degree, r)

        if sp.isspmatrix(adj):
            adj = adj.tocoo(copy=False)
            adj.data = degree_power[adj.row] * adj.data
            if symmetric:
                adj.data *= degree_power[adj.col]
            adj = adj.tocsr(copy=False)
        else:
            degree_power_matrix = sp.diags(degree_power)
            adj = degree_power_matrix @ adj
            if symmetric:
                adj = adj @ degree_power_matrix
        return adj

    if isinstance(rate, list):
        return tuple(_normalize_adj(adj_matrix, r) for r in rate)
    else:
        return _normalize_adj(adj_matrix, rate)


def precalculate_pe(data: Data, nb_pos_enc: int):
    N = data.num_nodes
    
    edge_index = to_undirected(data.edge_index, num_nodes=N)
    row, col = edge_index
    
    edge_index_norm, edge_weight_norm = get_laplacian(edge_index, normalization='sym')
    A = to_torch_csr_tensor(edge_index_norm, edge_weight_norm, size=data.size())
    mat = A @ A @ data.x
    
    # Notice: The propagation operator in the original repository of DRSGNN seems to be calculated by the 
    # element-wise multiplication between a normalized adjacency matrix \hat{A} and its inverse degree matrix D^{-1}.
    # In our implementation, we calculate the RWPE proposed in https://arxiv.org/abs/2110.07875.
    deg = torch.ones(data.num_edges, device=row.device)
    deg = scatter(deg, row, dim_size=N, reduce='sum').clamp(min=1)[row]
    deg = 1.0 / deg
    
    rw_opt = to_torch_csr_tensor(edge_index, deg, size=data.size())
    out = rw_opt
    
    pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=N)]
    for _ in tqdm(range(nb_pos_enc - 1)):
        out = out @ rw_opt
        pe_list.append(get_self_loop_attr(*to_edge_index(out), N))
    pe = torch.stack(pe_list, dim=-1)
    return mat, pe


def eliminate_selfloops(adj_matrix):
    """eliminate selfloops for adjacency matrix.

    >>>eliminate_selfloops(adj) # return an adjacency matrix without selfloops

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array

    Returns
    -------
    Single Scipy sparse matrix or Numpy matrix.

    """
    if sp.issparse(adj_matrix):
        adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format='csr')
        adj_matrix.eliminate_zeros()
    else:
        adj_matrix = adj_matrix - np.diag(adj_matrix)
    return adj_matrix


def add_selfloops(adj_matrix: sp.csr_matrix):
    """add selfloops for adjacency matrix.

    >>>add_selfloops(adj) # return an adjacency matrix with selfloops

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array

    Returns
    -------
    Single sparse matrix or Numpy matrix.

    """
    adj_matrix = eliminate_selfloops(adj_matrix)

    return adj_matrix + sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='csr')


def merge(edges, step=1):
    if step == 1:
        return edges
    i = 0
    length = len(edges)
    out = []
    while i < length:
        e = edges[i:i + step]
        if len(e):
            out.append(np.hstack(e))
        i += step
    print(f'Edges has been merged from {len(edges)} timestamps to {len(out)} timestamps')
    return out


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


def gcn_norm_with_k_order(data, name:str, K: int, root: str='./adj_normalized') -> List:
    adj_gcn_norm = []
    for time_step in tqdm(range(len(data))):
        snapshot = data[time_step]
        x, edge_index = snapshot.x, snapshot.edge_index
        
        adj = to_torch_sparse_tensor(edge_index, size=x.size(0))
        adj_normlized, _ = gcn_norm(adj, num_nodes=x.size(0))
        
        adj_tmp = adj_normlized.detach()
        for _ in range(K - 1):
            adj_normlized = mm(adj_normlized, adj_tmp)
        adj_gcn_norm.append(adj_normlized)
        
    torch.save(adj_gcn_norm, osp.join(root, f'{name}_K_{K}.pt'))
    return adj_gcn_norm