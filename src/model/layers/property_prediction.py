import torch
import torch.nn as nn
from torch import FloatTensor
from .block.fc import Linear

class PropertyPredictionModel(nn.Module):
    def __init__(self, 
                 core_graph_vector_dim: int,
                 property_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.0
                 ) -> None:
        super(PropertyPredictionModel, self).__init__()

        self.mlp = nn.Sequential(
            Linear(
                input_dim = core_graph_vector_dim,
                output_dim = hidden_dim,
                activation='relu',
                dropout=dropout
            ),
            Linear(
                input_dim = hidden_dim,
                output_dim = hidden_dim,
                activation='relu',
                dropout=dropout
            ),
            Linear(
                input_dim = hidden_dim,
                output_dim = hidden_dim,
                activation=None,
            )
        )

    def forward(self, Z_core: FloatTensor):
        return self.mlp(Z_core)