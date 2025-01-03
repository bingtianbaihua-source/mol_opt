import torch
from torch import nn
from torch import FloatTensor, LongTensor
from typing import Optional
from .fc import Linear
from torch_scatter import scatter_sum, scatter_mean, scatter_max


import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_mean, scatter_max

class Readout(nn.Module):
    """
    Input
    nodes : n_node, node_dim

    Output
    retval : n_graph, output_dim
    """
    def __init__(
        self,
        node_dim: int,
        output_dim: int,
        activation: Optional[str] = None,
        dropout: float = 0.0,
    ):
        super(Readout, self).__init__()

        self.linear1 = nn.Linear(node_dim, output_dim)
        self.linear2 = nn.Linear(node_dim, output_dim)
        self.activation = nn.ReLU(inplace=True) if activation is None else getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, node2graph: OptTensor = None) -> Tensor:
        """
        x: [V, F]
        node2graph: optional, [V, ]
        """
        if node2graph is not None:
            avg_pool = scatter_mean(x, node2graph, dim=0)  # V, F -> N, F
            max_pool = scatter_max(x, node2graph, dim=0)[0] # V, F -> N, F
        else:  # when N = 1
            avg_pool = x.mean(dim=0, keepdim=True)        # V, F -> 1, F
            max_pool = x.max(dim=0, keepdim=True)[0]      # V, F -> 1, F
        
        # Concatenate the average and max pool results
        pooled = torch.cat([avg_pool, max_pool], dim=-1)  # N, 2*F

        # Apply the linear layers, activation, and dropout
        out = self.linear1(pooled)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out
