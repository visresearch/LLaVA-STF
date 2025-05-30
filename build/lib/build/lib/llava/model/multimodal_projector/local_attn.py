import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp
import re
import copy
from typing import Optional

class LocalAttention(nn.Module):
    def __init__(self, dim, num_heads=32, mlp_ratio=4, qkv_bias=False, norm_layer=nn.LayerNorm, act_layer=nn.GELU, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.post_norm = norm_layer(dim, eps=1e-5)

    def forward(self, x):
        x_shortcut = x
        b, m ,n, c = x.shape
        qkv = self.qkv(x).reshape(b, m, n, 3, self.num_heads, c // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0) #b, h, m, n, c // h

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(b, m, n, c)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.post_norm(x)
        x = x + x_shortcut
        # x = self.post_norm(x)
        # x = x_shortcut + self.post_norm(x)
        x = x + self.post_norm(self.mlp(x))
        return x.reshape(b, m, n*c)

