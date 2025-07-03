from multiprocessing import context
from re import X
from numpy import e
import torch
from torch import nn
from ..submodule.SelfAttention import MultiHeadAttention
from ..submodule.NormLayer import NormLayer
from ..submodule.FeedForward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.norm1 = NormLayer(cfg["embedding_dim"])
        self.norm2 = NormLayer(cfg["embedding_dim"])
        self.att = MultiHeadAttention(
            d_in=cfg["embedding_dim"],
            d_out=cfg["embedding_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(embedding_dim=cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x += shortcut
        return x
