import torch.nn as nn

from .conv import RiemannianSGNNConv, RSEncoderConv
from .manifolds import Lorentz


class RiemannianSpikeGNN(nn.Module):
    def __init__(
        self, in_channels, out_channels, manifold, T, num_layers, step_size, embed_dim,
        v_threshold=1.0, dropout=0.1, neuron="IF", delta=0.05, 
        tau=2, use_MS=True
    ):
        super(RiemannianSpikeGNN, self).__init__()
        if isinstance(manifold, Lorentz):
            embed_dim += 1
        self.manifold = manifold
        self.step_size = step_size
        self.encoder = RSEncoderConv(manifold, T, in_channels, embed_dim, neuron=neuron, delta=delta, tau=tau,
                                      step_size=step_size, v_threshold=v_threshold,
                                      dropout=dropout, use_MS=use_MS)
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(
                RiemannianSGNNConv(manifold, embed_dim, neuron=neuron, delta=delta, tau=tau,
                                                       step_size=step_size, v_threshold=v_threshold,
                                                       dropout=dropout, use_MS=use_MS)
                                   )
        self.fc = nn.Linear(embed_dim, out_channels, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        x, z, y = self.encoder(x, edge_index)
        v = y.clone()
        for layer in self.layers:
            x, z, y = layer(x, z, edge_index)
            v += y

        z1 = self.manifold.proju0(self.manifold.logmap0(z))
        z1 = self.fc(z1)
        
        if self.training:
            return z1
        else:
            return self.fc(self.manifold.proju0(v))

    def infer(self, x, edge_index, edge_weight=None):
        x, y = self.encoder.infer(x, edge_index)
        v = y.clone()
        for layer in self.layers:
            x, y = layer.infer(x, edge_index)
            v += y
        
        return self.fc(self.manifold.proju0(v))