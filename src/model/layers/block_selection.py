import torch
import torch.nn as nn
from torch import FloatTensor
from .block.fc import Linear

class BlockSelectionModel(nn.Module):
    def __init__(self, 
                 core_graph_vector_dim: int,
                 block_graph_vector_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.0
                 ) -> None:
        super(BlockSelectionModel, self).__init__()

        self.mlp = nn.Sequential(
            Linear(
                input_dim=core_graph_vector_dim + block_graph_vector_dim,
                output_dim=hidden_dim,
                activation='relu',
                dropout=dropout
            ),
            Linear(
                input_dim=hidden_dim,
                output_dim=1,
                activation='sigmoid'
            )
        )
        
    def forward(self,
                Z_core: FloatTensor,
                Z_block: FloatTensor
                ):
        Z_concat = torch.cat([Z_core, Z_block], dim=-1)
        return self.mlp(Z_concat).sequeeze(-1)