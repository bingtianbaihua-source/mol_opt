from torch import nn
from .activation import ShiftedSoftplus
from typing import Optional

__all__ = [
    'Linear',
]

ACT_LIST = {
    'relu' : nn.ReLU(),
    'Relu' : nn.ReLU(),
    'ReLU' : nn.ReLU(),
    'silu' : nn.SiLU(),
    'SiLu' : nn.SiLU(),
    'SiLU' : nn.SiLU(),
    'tanh' : nn.Tanh(),
    'sigmoid' : nn.Sigmoid(),
    'Sigmoid' : nn.Sigmoid(),
    'ShiftedSoftplus' : ShiftedSoftplus(),
    'shiftedsoftplus' : ShiftedSoftplus(),
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

        # 预先创建 Identity 实例
        identity = nn.Identity()
        
        activation_layer = ACT_LIST[activation] if activation is not None else identity
        norm_layer = NORM_LIST[norm](output_dim) if norm is not None else identity
        
        super(Linear, self).__init__(
            nn.Linear(input_dim, output_dim, bias=bias),
            norm_layer,
            activation_layer,
            nn.Dropout(p=dropout)
        )
