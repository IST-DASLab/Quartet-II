import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform

from models.quantization.quantizers.base import BaseQuantizer
from .nvfp4_triton import rtn_1x16s_fp4_autograd, rtn_16x16s_fp4_autograd


def rtn_fp4(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    inds = torch.bucketize(x, grid)

    lo = torch.clamp(inds - 1, min=0, max=15)
    hi = torch.clamp(inds,     min=0, max=15)

    low = grid[lo]
    high = grid[hi]

    return torch.where(
        (high - x) <= (x - low),
        high,
        low,
    )


class Nvfp4Quantizer(BaseQuantizer):
    grid = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0, 6.0],
        device="cuda",
    )
    
    def __init__(self, hadamard_dim=1, square: bool=True, scale_override: float=1.0):
        super().__init__(4)
        
        self.hadamard_dim = hadamard_dim
        if self.hadamard_dim != 1:
            self.hadamard_matrix = hadamard_transform(
                torch.eye(hadamard_dim, dtype=torch.float32, device="cuda"), scale=hadamard_dim**-0.5
            )
        self.square = square
        self.scale_override = scale_override

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hadamard_dim={self.hadamard_dim}, "
            f"square={self.square}, "
            f"scale_override={self.scale_override})"
        )
        
    def round_scales(self, scales):            
        s_dec = scales.max() / (447.99 * 6.0)
        s_dec[s_dec == 0] = 1.0
        s_dec_b = scales / 6.0
        s_dec_b_e4m3 = (s_dec_b / s_dec).to(torch.float8_e4m3fn).float()
        s_dec_b_e4m3[s_dec_b_e4m3 == 0] = 1.0
        s_enc_b_inv = s_dec_b_e4m3 * s_dec
        return s_enc_b_inv

    def forward(self, x):
        if hasattr(self, "hadamard_matrix"):
            self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        self.grid = self.grid.to(x.device).to(x.dtype)
        
        if (
            self.hadamard_dim == 1 and
            not self.square
        ):
            return rtn_1x16s_fp4_autograd.apply(x, self.scale_override, 16)
        elif (
            self.hadamard_dim == 1 and
            self.square
        ):
            return rtn_16x16s_fp4_autograd.apply(x, self.scale_override, 16)
            
        
        if self.hadamard_dim != 1:
            x_had = F.linear(x.view(-1, self.hadamard_dim), self.hadamard_matrix).view_as(x)
        else:
            x_had = x.clone()

        with torch.no_grad():  
            if self.square:
                x_grouped = x_had.view(x.shape[0] // 16, 16, x.shape[1] // 16, 16).permute(0, 2, 1, 3).reshape(-1, 16 * 16)
            else:
                x_grouped = x_had.view(-1, 16)
            
            scales = x_grouped.abs().max(dim=-1, keepdim=True)[0]
            s_enc_b_inv = self.round_scales(scales)
            x_fp4 = rtn_fp4(x_grouped / s_enc_b_inv, self.grid) * s_enc_b_inv
            
            if self.square:    
                x_fp4 = x_fp4.reshape(x.shape[0] // 16, x.shape[1] // 16, 16, 16).permute(0, 2, 1, 3).reshape_as(x)
            else:
                x_fp4 = x_fp4.view_as(x)

        if self.hadamard_dim != 1: 
            return (x_had + (x_fp4 - x_had).detach())
        else:
            return x + (x_fp4 - x).detach()


class QuestNvfp4Quantizer(BaseQuantizer):
    grid = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0, 6.0],
        device="cuda",
    )
    
    def __init__(self, hadamard_dim=16, scale_override: float=1.0):
        super().__init__(4)
        
        self.hadamard_dim = hadamard_dim
        if self.hadamard_dim != 1:
            self.hadamard_matrix = hadamard_transform(
                torch.eye(hadamard_dim, dtype=torch.float32, device="cuda"), scale=hadamard_dim**-0.5
            )
        self.scale_override = scale_override

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hadamard_dim={self.hadamard_dim}, "
            f"scale_override={self.scale_override})"
        )
        
    def round_scales(self, scales):            
        global_scale = scales.max() / 256.0
        scales = scales / global_scale
        scales = scales.to(torch.float8_e4m3fn).float()
        scales = torch.where(
            scales == 0.0,
            1.0,
            scales,
        )
        return scales, global_scale * self.scale_override

    def forward(self, x):
        if hasattr(self, "hadamard_matrix"):
            self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        self.grid = self.grid.to(x.device).to(x.dtype)
        
        if self.hadamard_dim != 1:
            x_had = F.linear(x.view(-1, self.hadamard_dim), self.hadamard_matrix).view_as(x)
        else:
            x_had = x

        with torch.no_grad():  
            x_grouped = x_had.view(-1, 16)
            
            scales = x_grouped.pow(2).mean(dim=-1, keepdim=True).pow(1/2) * (2.92247856 / 6.0)
            scales, global_scale = self.round_scales(scales)
            x_scaled = x_grouped / scales / global_scale
            x_fp4 = rtn_fp4(x_scaled, self.grid) * scales * global_scale
            
            x_fp4 = x_fp4.view_as(x)

        return (x_had + (x_fp4 - x_had).detach())
