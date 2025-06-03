from .stfn_dyn import STFN
from .spikinggcn_dyn import SpikingGCN
from .spikegt_dyn import SpikeGT
from .signn_dyn import SiGNN
from .spikenet_dyn import SpikeNet


__all__ = [
    'SiGNN',
    'STFN',
    'SpikingGCN',
    'SpikeNet',
    'SpikeGT',
]


def load_model(data, args):
    if args.model_name == 'stfn':
        model = STFN(
            data, data.num_features, data.num_classes, args.hids, args.sizes,
            args.neuron, args.v_threshold, args.p, dropout=args.dropout
        )
    elif args.model_name == 'spikinggcn':
        model = SpikingGCN(
            data, data.num_features, data.num_classes, args.hid_channels, args.K, 
            args.neuron, args.v_threshold, args.tau, dropout=args.dropout
        )
    elif args.model_name == 'spikegt':
        model = SpikeGT(
            data, data.num_features, data.num_classes, args.hid_channels, args.hids, args.sizes, 
            args.num_heads, dropout=args.dropout, graph_weight=args.graph_weight, depths=args.depths
        )
    elif args.model_name == 'spikenet':
        model = SpikeNet(
            data, data.num_features, data.num_classes, args.hids, args.sizes,
            p=args.p, dropout=args.dropout, neuron=args.neuron
        )
    elif args.model_name == 'signn':
        model = SiGNN(data, data.num_features, data.num_classes, args.hids, args.sizes, p=args.p, dropout=args.dropout)
    else:
        raise NotImplementedError

    return model