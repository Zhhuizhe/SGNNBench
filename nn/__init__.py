from .stfn import STFN
from .spikinggcn import SpikingGCN
from .spikegt import SpikeGraphTransformer
from .spikenet import SpikeNet
from .signn import SiGNN
from .spikegcl import SpikeGCL
from .msg import RiemannianSpikeGNN
from .drsgnn import DRSGNN

from .mpnn import MPNN

from .manifolds import Euclidean, Lorentz, Sphere, ProductSpace


__all__ = [
    'SiGNN',
    'STFN',
    'SpikingGCN',
    'SpikeNet',
    'SpikeGraphTransformer',
    'SpikeGCL',
    'MPNN',
]


def load_model(name:str, in_channels:int, out_channels: int, model_kwargs):
    model = None
    if name.lower() == 'stfn':
        model = STFN(in_channels, out_channels, **model_kwargs)
    elif name.lower() == 'spikinggcn':
        model = SpikingGCN(in_channels, out_channels, **model_kwargs)
    elif name.lower() == 'spikegt':
        model = SpikeGraphTransformer(in_channels, out_channels, **model_kwargs)
    elif name.lower() == 'spikegcl':
        model = SpikeGCL(in_channels, **model_kwargs)
    elif name.lower() == 'mpnn':
        model = MPNN(in_channels, out_channels, **model_kwargs)
    elif name.lower() == 'msg':
        m_tuple = []
        for i, m_str in enumerate(model_kwargs.manifold):
            if m_str == "euclidean":
                m_tuple.append((Euclidean(), model_kwargs.embed_dim[i]))
            elif m_str == 'lorentz':
                m_tuple.append((Lorentz(), model_kwargs.embed_dim[i]))
            elif m_str == 'sphere':
                m_tuple.append((Sphere(), model_kwargs.embed_dim[i]))

        if model_kwargs.use_product:
            print('using product space')
            manifold = ProductSpace(*m_tuple)
        else:
            manifold = m_tuple[0][0]
        model = RiemannianSpikeGNN(
            in_channels, out_channels, manifold, model_kwargs.T,
            model_kwargs.num_layers, model_kwargs.step_size, sum(model_kwargs.embed_dim),
            model_kwargs.v_threshold, model_kwargs.dropout, model_kwargs.neuron,
            model_kwargs.delta, model_kwargs.tau, model_kwargs.use_MS,
        )
    elif name == 'drsgnn':
        model = DRSGNN(in_channels, out_channels, model_kwargs.T)
    else:
        raise NotImplementedError

    return model