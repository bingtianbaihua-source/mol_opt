from torch import Tensor
from torch import nn
from torch.nn import functional as F
import math

def shiftedsoftplus(input: Tensor):
    return F.softplus(input) - math.log(2.0)

class ShiftedSoftplus(nn.Module):

    def forward(self, input: Tensor):
        return shiftedsoftplus(input)