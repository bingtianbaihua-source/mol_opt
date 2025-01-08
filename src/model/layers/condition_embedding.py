import torch
import torch.nn as nn
from torch import FloatTensor
from .block.fc import Linear
from typing import Optional

class ConditionEmbeddingModel(nn.Module):
    def __init__(self, 
                 core_node_vector_dim: int,
                 core_graph_vector_dim: int,
                 condition_dim: int,
                 dropout: float = 0.0
                 ) -> None:
        super(ConditionEmbeddingModel, self).__init__()
        self.node_mlp = nn.Sequential(
            Linear(
                input_dim=core_node_vector_dim + condition_dim,
                output_dim=core_node_vector_dim,
                activation='SiLU',
                dropout=dropout
            ),
            Linear(
                input_dim=core_node_vector_dim,
                output_dim=core_node_vector_dim,
                activation='SiLU',
                dropout=dropout
            ),
        )
        self.graph_mlp = nn.Sequential(
            Linear(
                input_dim = core_graph_vector_dim + condition_dim,
                output_dim= core_graph_vector_dim,
                activation='SiLU',
                dropout=dropout
            ),
            Linear(
                input_dim = core_graph_vector_dim,
                output_dim= core_graph_vector_dim,
                activation='SiLU',
                dropout=dropout
            )
        )

    def forward(self,
                h_core: FloatTensor,
                Z_core: FloatTensor,
                condition: FloatTensor,
                node2graph: Optional[FloatTensor]):
        
        if node2graph is not None:
            h_condition = condition[node2graph]
        else:
            h_condition = condition.repeat(h_core.size(0), 1)
        Z_condition = condition
        h_core = torch.cat([h_core, h_condition], dim=-1)
        Z_core = torch.cat([Z_core, Z_condition], dim=-1)
        return self.node_mlp(h_core), self.graph_mlp(Z_core)