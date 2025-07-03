import torch
import torch.nn as nn
from .GELU import GELU


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.layers(x)
