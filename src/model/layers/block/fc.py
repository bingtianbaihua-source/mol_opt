from torch import nn
from .activation import ShiftedSoftplus
from typing import Optional

__all__ = [
    'Linear',
]

ACT_LIST = {
    'relu' : nn.ReLU,
    'Relu' : nn.ReLU,
    'silu' : nn.SiLU,
    'SiLu' : nn.SiLU,
    'tanh' : nn.Tanh,
    'sigmoid' : nn.Sigmoid,
    'Sigmoid' : nn.Sigmoid,
    'ShiftedSoftplus' : ShiftedSoftplus,
    'shiftedsoftplus' : ShiftedSoftplus,
}

NORM_LIST = {
    'LayerNorm' : nn.LayerNorm,
    'BatchNorm' : nn.BatchNorm1d,
}

class Linear(nn.Sequential):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 activation: Optional[str] = None,
                 norm: Optional[str] = None,
                 bias: bool = True,
                 dropout: float = 0.0):

        activation_layer = ACT_LIST[activation] if activation is not None else nn.Identity()
        norm_layer = NORM_LIST[norm] if norm is not None else nn.Identity()
        super(Linear, self).__init__(
            nn.Linear(input_dim, output_dim, bias=bias),
            norm_layer,
            activation_layer,
            nn.Dropout(p=dropout)
        )
