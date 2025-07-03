import torch
from torch import nn
import torch.nn.functional as F

class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        __init__ 初始化自注意力模块

        Args:
            d_in (int): 嵌入维度
            d_out (int): KQV维度
            qkv_bias (bool, optional): _description_. Defaults to False.
        """
        super(SelfAttentionV2, self).__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        forward _summary_

        Args:
            x (_type_): x.shape=context_length*d_in

        Returns:
            _type_: _description_
        """
        keys = self.W_key(x)
        querys = self.W_query(x)
        values = self.W_value(x)
        attention_scores = querys @ keys.T

        # 使用掩码机制实现因果注意力
        mask = torch.tril(
            torch.ones(attention_scores.shape[0], attention_scores.shape[0])
        )
        masked_scores = attention_scores.masked_fill(mask == 0, -torch.inf)
        attention_weights = torch.softmax(masked_scores / keys.shape[-1] ** 0.5, dim=-1)

        # 上下文矩阵
        context_matrix = attention_weights @ values
        return context_matrix

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        __init__ 初始化自注意力模块

        Args:
            d_in (int): 嵌入维度
            d_out (int): KQV维度
            context_length (int): 上下文长度
            dropout (float): 丢弃概率
            qkv_bias (bool, optional): _description_. Defaults to False.
        """
        super(CasualAttention, self).__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        forward _summary_

        Args:
            x (_type_): x.shape=batch_size * context_length * d_in

        Returns:
            _type_: _description_
        """
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        querys = self.W_query(x)
        values = self.W_value(x)
        attention_scores = querys @ keys.transpose(
            1, 2
        )  # 将context_length*d_in进行转置，保持batch_size在第0个维度

        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.Dropout(attention_weights)
        # 上下文矩阵
        context_matrix = attention_weights @ values
        return context_matrix

# 简单的多头注意力类封装
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, num_heads, d_in, d_out, context_length, dropout, qkv_bias=False):
        super(MultiHeadAttentionWrapper, self).__init__()
        self.heads = nn.ModuleList(
            [
                CasualAttention(
                    d_in,
                    int(d_out / num_heads),
                    context_length,
                    dropout,
                    qkv_bias=False,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_in, d_out, context_length, dropout, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 取整除法
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        ###将keys value queries维度转变为 b*num_heads*num_tokens*head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)
        ###
        
        attention_scores = queries @ keys.transpose(-2, -1)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill_(mask_bool, -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.head_dim**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_matrix = (attention_weights @ values).transpose(1, 2)
        context_matrix = context_matrix.contiguous().view(b, num_tokens, self.d_out)
        context_matrix = self.out_proj(context_matrix)  # 添加一个可选的线性投影
        return context_matrix
