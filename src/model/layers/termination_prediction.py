import torch
import torch.nn as nn
from torch import FloatTensor
from .block.fc import Linear

class TerminationPredictionModel(nn.Module):
    def __init__(self, 
                 core_graph_vector_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.0
                 ) -> None:
        super(TerminationPredictionModel, self).__init__()
        self.mlp = nn.Sequential(
            Linear(
                input_dim=core_graph_vector_dim,
                output_dim=hidden_dim,
                activation='ReLU',
                dropout=dropout
            ),
            Linear(
                input_dim=hidden_dim,
                output_dim=1,
                activation=None
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, Z_core: FloatTensor, return_logit = False):
        logit = self.mlp(Z_core).squeeze(-1)
        if return_logit is False:
            return self.sigmoid(logit)
        else:
            return logit