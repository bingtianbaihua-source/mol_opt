import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import math


'''
根据距离进行平滑衰减，在距离接近cutoff时将权重逐渐减小到0
'''
def cosinecutoff(input: Tensor, cutoff: float):
    return 0.5 * (torch.cos(math.pi * input / cutoff))

class CosineCutoff(nn.Module):
    def __init__(self, cutoff = 10.0):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, edge_distance):
        return cosinecutoff(edge_distance, self.cutoff)

