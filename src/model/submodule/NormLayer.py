import torch
from torch import nn


class NormLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(NormLayer, self).__init__()
        self.eps = 1e-6
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x + self.shift
