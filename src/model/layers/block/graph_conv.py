from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from typing import Union, Tuple, Optional
from .fc import *
import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import LayerNorm

class ResidualBlock(nn.Module):
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int,
                 activation: Optional[str] = 'SiLU',
                 layer_norm: Optional[str] = None,
                 dropout: float = 0.0,
                 ) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = GraphConv(node_dim, edge_dim, activation, layer_norm, dropout)
        self.graph_norm1 = LayerNorm(in_channels=node_dim, mode='graph')
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = GraphConv(node_dim, edge_dim, activation, layer_norm, dropout)
        self.graph_norm2 = LayerNorm(in_channels=node_dim, mode='graph')
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self,
                x: Tensor, 
                edge_index: Adj, 
                edge_attr: Tensor,
                node2graph: OptTensor,
                ):
        identity = x

        out = self.conv1(x, edge_index, edge_attr = edge_index)
        out = self.graph_norm1(out, node2graph)
        out = self.relu1(out)
        out = self.conv2(out, edge_index, edge_attr = edge_index)
        out = (out + identity) / 2
        
        return self.relu2(out)

class GraphConv(MessagePassing):
    def __init__(self,
                 node_dim: Union[int, Tuple[int]],
                 edge_dim: int,
                 activation: Optional[str] = None,
                 norm: Optional[str] = None,
                 dropout: float = 0.0,
                 **kwargs,
                 ):
        super(GraphConv, self).__init__(**kwargs)

        if isinstance(node_dim, int):
            src_node_dim, dst_node_dim = node_dim, node_dim
        else:
            src_node_dim, dst_node_dim = node_dim

        self.edge_layer = Linear(edge_dim, src_node_dim, activation, dropout = dropout)
        self.eps = torch.nn.parameter(torch.Tensor([0.1]))
        self.nn = nn.Sequential(
            Linear(dst_node_dim, dst_node_dim, activation=activation, norm=norm, dropout=dropout),
            Linear(dst_node_dim, dst_node_dim, activation=activation, norm=norm),
        )

    def forward(self,
                x: Union[Tensor, OptPairTensor], 
                edge_index: Adj,
                edge_attr: Tensor) -> Tensor:
        
        r'''
        x: node feature 
        edge_index: edge index       (2, E)
        edge_attr: edge feature      (E, Fe)
        '''

        if isinstance(x, Tensor):
            x: OptPairTensor = (x,x)
        x_src, x_dst = x

        x_dst_update = self.propagate(edge_index, x = x, edge_attr = edge_attr)

        # 带参数的残差连接
        if x_dst is not None:
            x_dst_update = x_dst_update + (1 + self.eps) * x_dst

        return self.nn(x_dst_update)
    
    def message(self, x_j, edge_attr):
        edge_attr = self.edge_layer(edge_attr)
        return (x_j + edge_attr).relu()
        