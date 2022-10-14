import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import math

from src.utils import clones

class EncoderBlock(nn.Module):
    def __init__(self, dropout=0.1, d_model=240, d_ff=128, h=8):
        super(EncoderBlock, self).__init__()
        self.mod = torch.nn.Linear(1, d_model)
        self.linear1 = torch.nn.Linear(d_model, d_model)
        self.att = torch.nn.MultiheadAttention(d_model, h)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = x.float()
        x = self.sublayer[0](x, lambda x: self.att(x, x, x)[0])
        return self.sublayer[1](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len= 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeFilm(nn.Module):
    def __init__(self, n_harmonics=7, embedding_size=64, T_max=1000.0, input_size = 1):
        super(TimeFilm, self).__init__()

        self.a = nn.parameter.Parameter(
            torch.rand(n_harmonics, embedding_size), requires_grad=True)
        self.b = nn.parameter.Parameter(
            torch.rand(n_harmonics, embedding_size), requires_grad=True)
        self.w = nn.parameter.Parameter(
            torch.rand(n_harmonics, embedding_size), requires_grad=True)
        self.v = nn.parameter.Parameter(
            torch.rand(n_harmonics, embedding_size),  requires_grad=True)

        self.linear_proj = nn.Sequential(nn.Linear(in_features= input_size, out_features=embedding_size, bias=False),
                                         nn.LeakyReLU(0.1))

        self.linear_proj_ = nn.Sequential(nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=False),
                                          nn.LeakyReLU(0.1))
        self.n_ = nn.parameter.Parameter(
            torch.linspace(1, n_harmonics+1, steps=n_harmonics) / T_max, requires_grad=False)

    def harmonics(self, t):
        """ t [n_batch, length sequence, 1, n_harmonics]"""

        return t[:, :, :, None]*2*np.pi*self.n_

    def fourier_coefs(self, t):

        t_harmonics = self.harmonics(t)

        gama_ = torch.tanh(torch.matmul(torch.sin(t_harmonics), self.a) + \
            torch.matmul(torch.cos(t_harmonics), self.b))

        beta_ = torch.matmul(torch.sin(t_harmonics), self.v) + \
            torch.matmul(torch.cos(t_harmonics), self.w)

        return gama_, beta_

    def forward(self, x, t):
        """ t must be of size [n_batch, length sequence]"""

        gama_, beta_ = self.fourier_coefs(t)

        # self.linear_proj_(self.linear_proj(x[:, :, None])*torch.tanh(torch.squeeze(gama_)) + torch.squeeze(beta_))
        return self.linear_proj_(self.linear_proj(x)*torch.squeeze(gama_) + torch.squeeze(beta_))        


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len= 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


class PositionalEncodingSousa(nn.Module):
    "Implement the PE function."
    def __init__(self, d_TE, maxtime, dropout, max_len=5000):
        super(PositionalEncodingSousa, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len, d_TE)
        self.div_term = torch.exp(torch.arange(0, d_TE, 2) *
                             -(math.log(maxtime) / d_TE))
        
    def forward(self, x, time):
        position = torch.arange(0, time).unsqueeze(1)
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        self.pe[:, 0::2] = torch.sin(position * self.div_term)
        self.pe[:, 1::2] = torch.cos(position * self.div_term)
        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('pe', self.pe)
        
        return self.dropout(x)
