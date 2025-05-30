import torch
import torch.nn as nn
import re
from .local_attn import LocalAttention

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def build_local_attention():
    return LocalAttention(dim=1024)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"projector type: {projector_type}")
    if projector_type == 'linear':
        return nn.Linear(config.hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        # modules = [nn.Linear(config.hidden_size, config.hidden_size*4)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        # modules.append(nn.GELU())
        # modules.append(nn.Linear(config.hidden_size*4, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_projector_crosslayer(config, delay_load=False, **kwargs):

    # return nn.Linear(config.mm_hidden_size*2, config.mm_hidden_size)

    modules = [nn.Linear(config.mm_hidden_size*8, config.mm_hidden_size*4)]
    modules.append(nn.GELU())
    modules.append(nn.Linear(config.mm_hidden_size*4, config.mm_hidden_size))
    # modules.append(nn.GELU())
    # modules.append(nn.Linear(config.hidden_size*4, config.hidden_size))
    return nn.Sequential(*modules)

def build_vision_projector_neighborlayer(config, delay_load=False, **kwargs):

    return nn.Linear(config.mm_hidden_size*3, config.mm_hidden_size)

    # modules = [nn.Linear(config.mm_hidden_size*4, config.mm_hidden_size*4)]
    # modules.append(nn.GELU())
    # modules.append(nn.Linear(config.mm_hidden_size*4, config.mm_hidden_size))
    # # modules.append(nn.GELU())
    # # modules.append(nn.Linear(config.hidden_size*4, config.hidden_size))
    # return nn.Sequential(*modules)

def build_vision_projector_light(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"projector type: {projector_type}")
    if projector_type == 'linear':
        return nn.Linear(config.hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        # modules = [nn.Linear(config.hidden_size, config.hidden_size*4)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.mm_hidden_size))
        # modules.append(nn.GELU())
        # modules.append(nn.Linear(config.hidden_size*4, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_linear_projector(config):
    return nn.Linear(config.hidden_size, config.hidden_size)

def build_vision_projector_baseline(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"projector type teachers: {projector_type}")
    if projector_type == 'linear':
        return nn.Linear(config.hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
