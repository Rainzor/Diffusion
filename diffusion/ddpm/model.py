# reference:
# https://github.com/lucidrains/denoising-diffusion-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert x.size(1) > self.pe.size(0), "Input sequence length exceeds positional encoding length."

        return x + self.pe[:x.size(1), :].unsqueeze(0)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(CNNBlock, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.RMSNorm(out_channels) # without scale parameter
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None: # compensate scale and shift
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.cnn1 = CNNBlock(in_channels, out_channels, dropout)
        self.cnn2 = CNNBlock(out_channels, out_channels, dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, scale_shift=None):
        residual = self.shortcut(x)
        x = self.cnn1(x, scale_shift)
        x = self.cnn2(x, scale_shift)
        return x + residual
