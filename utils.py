import yaml
import random
import logzero
import scipy.sparse as sp
import os.path as osp
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from texttable import Texttable

import torch
import torch_cluster
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix

from sample_neighber import sample_neighber_cpu


HOMO_DATASETS = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CS', 'Physics']
HETERO_DATASETS = ['Actor', 'Squirrel', 'Chameleon', 'Amazon-ratings', 'Roman-empire', 'Minesweeper', 'Questions']
LARGE_DATASETS = ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_model_params(model):
    """
    Return the size of model in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2

    
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
    def load_config_from_file(cls, args_dir):
        assert osp.exists(args_dir), "Error: The configuration path is not found!"
        
        _, file_extention = osp.splitext(args_dir)
        if file_extention in [".yml", ".yaml"]:
            with open(args_dir, "r") as cfg_file:
                cfg_as_dict = yaml.safe_load(cfg_file)
                return cls(cfg_as_dict)
        else:
            raise NotImplementedError
    
    @classmethod
    def _create_config_tree_from_dict(cls, init_dict):
        dic = deepcopy(init_dict)
        
        for (key, value) in dic.items():
            if isinstance(value, dict):
                dic[key] = cls(value)
        return dic
    
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)


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


def precalculate_pe(data, nb_pos_enc):
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    
    A = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes)
    A = normalize_adj(A)
    
    X = data.x.numpy()
    mat = A@A@X

    # compute the degree matrix
    d = np.zeros([len(X)])
    ind = np.where(A.todense() != 0)[0]
    for i in tqdm(range(len(X))):
        d[i] = len(np.where(ind == i)[0])
    Dinv = sp.diags(d ** -1.0, dtype=float)
    RW = A * Dinv
    M = RW

    # Iterate
    PE = [M.diagonal().astype(float)]
    M_power = M
    for _ in tqdm(range(nb_pos_enc - 1)):
        M_power = M_power * M
        PE.append(M_power.diagonal().astype(float))
    PE = np.stack(PE, axis=-1)
    return mat, PE


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