import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform

from models.quantization.quantizers.base import BaseQuantizer
from .nvfp4_triton import sr_1x16s_fp4_kernel_wrapper, eden_1x16s_fp4_kernel_wrapper, rtn_1x16s_fp4_kernel_wrapper


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


def sr_fp4(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # if (x.abs() > 6.001).any():
    #     raise ValueError(f"Can't SR overflowing tensor: {x.abs().max().item()} > 6")
    x = torch.clamp(x, -6.0, 6.0)
    if x.isnan().any():
        raise ValueError("x has NaNs")
    
    inds = torch.bucketize(x, grid)

    lo = torch.clamp(inds - 1, min=0, max=15)
    hi = torch.clamp(inds,     min=0, max=15)
    
    hi = torch.where(
        hi == lo,
        lo + 1,
        hi,
    )

    low = grid[lo]
    high = grid[hi]

    return torch.where(
        torch.bernoulli(
            (x - low) / (high - low)
        ) == 1.0,
        high,
        low,
    )


def sr_e4m3(x: torch.Tensor) -> torch.Tensor:
    # if (x > 448.001).any():
    #     raise ValueError(f"Can't SR overflowing tensor: {x.max().item()} > 448")
    x = torch.clamp(x, -447.99, 447.99) # insure 448 isnt low to prevent high from becoming NaN
    
    if x.isnan().any():
        raise ValueError("x has NaNs")
    
    q = x.to(torch.float8_e4m3fn)
    nextdq = (q.view(torch.uint8) + 1).view(torch.float8_e4m3fn).float()
    prevdq = (q.view(torch.uint8) - 1).view(torch.float8_e4m3fn).float()
    dq = q.float()

    low = torch.where(
        dq > x,
        prevdq,
        dq,
    )
    
    high = torch.where(
        dq > x,
        dq,
        nextdq,
    )
    
    return torch.where(
        torch.bernoulli(
            (x - low) / (high - low)
        ) == 1.0,
        high,
        low,
    )
    
    
def sr_e8m0(x: torch.Tensor) -> torch.Tensor:
    if x.isnan().any():
        raise ValueError("x has NaNs")
    
    low = 2 ** (torch.floor(torch.log2(x)))
    high = low * 2
    
    prob = (x - low) / (high - low)
    prob = torch.clamp(prob, 0, 1)
    
    return torch.where(
        torch.bernoulli(prob) == 1.0,
        high,
        low,
    )


class EdenSRQuantizer(BaseQuantizer):
    grid = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0, 6.0],
        device="cuda",
    )
    
    def __init__(self, hadamard_dim=32, group_dim=None, scale_dtype="fp32", unbiased="eden", rerotate=None, scale_override:float=1.0):
        super().__init__(4)
        
        self.hadamard_dim = hadamard_dim
        self.hadamard_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.float32, device="cuda"), scale=hadamard_dim**-0.5
        )
        if group_dim is None:
            group_dim = hadamard_dim
        self.group_dim = group_dim
        self.rerotate = rerotate
        self.scale_dtype = scale_dtype
        self.unbiased = unbiased
        self.scale_override = scale_override
        
        if scale_override != 1 and unbiased == "sr":
            raise ValueError("Scale Override is incompatible with Stochastic Rounding")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hadamard_dim={self.hadamard_dim}, "
            f"scale_dtype={self.scale_dtype}, "
            f"unbiased={self.unbiased}, "
            f"scale_override={self.scale_override}, "
            f"rerotate={self.rerotate})"
        )
        
    def round_scales(self, scales: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if self.scale_dtype == "fp32":
            return scales, torch.tensor([1.0 / 6.0 * self.scale_override], device=scales.device, dtype=scales.dtype)
        elif self.scale_dtype == "e4m3":
            global_scale = scales.max() / 256.0
            global_scale[global_scale == 0] = 1.0
            scales = scales / global_scale
            scales = scales.to(torch.float8_e4m3fn).float()
            scales[scales == 0] = 1.0
            return scales, global_scale / 6.0 * (17 / 16) * self.scale_override
            # s_dec = scales.max() / 447.99 * 6.0
            # s_dec[s_dec == 0] = 1.0
            # s_dec_b = scales / 6.0
            # s_dec_b_e4m3 = (s_dec_b / s_dec).to(torch.float8_e4m3fn).float()
            # s_dec_b_e4m3[s_dec_b_e4m3 == 0] = 1.0
            # return s_dec_b_e4m3, s_dec * self.scale_override
        elif self.scale_dtype == "e8m0":
            scales = 2 ** (torch.floor(torch.log2(scales)))
            return scales, torch.tensor([1 / 3.0 * self.scale_override], device=scales.device, dtype=scales.dtype)
        
    def apply_correction(self, scales: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        scales = scales.view(correction.size(0), -1)
        corrected_scales = (scales * correction).view(-1, 1)

        if self.scale_dtype == "fp32":
            return corrected_scales
        elif self.scale_dtype == "e4m3":
            # scales must remain E4M3 representable
            return sr_e4m3(corrected_scales)
        elif self.scale_dtype == "e8m0":
            # scales must remain E8M0 representable
            return sr_e8m0(corrected_scales)
        else:
            raise ValueError(f"Unknown scale_dtype: {self.scale_dtype}")
    
    def forward(self, x):
        self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        self.grid = self.grid.to(x.device).to(x.dtype)
        
        x_had = F.linear(x.view(-1, self.hadamard_dim), self.hadamard_matrix).view_as(x)
        if (
            self.scale_dtype == "e4m3" and
            self.unbiased == "eden"
        ):
            amax = torch.amax(x_had)
            return eden_1x16s_fp4_kernel_wrapper(x_had, (17 / 16) * self.scale_override, self.hadamard_dim, self.group_dim, amax)
        elif (
            self.scale_dtype == "e4m3" and
            self.unbiased == "sr"
        ):
            amax = torch.amax(x_had)
            return sr_1x16s_fp4_kernel_wrapper(x_had, (17 / 16) * self.scale_override, self.group_dim, amax)
        elif (
            self.scale_dtype == "e4m3" and
            self.unbiased == "no"
        ):
            amax = torch.amax(x_had)
            return rtn_1x16s_fp4_kernel_wrapper(x_had, (17 / 16) * self.scale_override, self.group_dim, amax)

        x_had = x_had.view(-1, self.group_dim)
        scales = x_had.abs().max(dim=-1, keepdim=True)[0]
        
        scales, global_scale = self.round_scales(scales)
        
        x_scaled = x_had / scales / global_scale
        if self.unbiased == "no":
            x_fp4 = rtn_fp4(x_scaled, self.grid)
        elif self.unbiased == "sr":
            x_fp4 = sr_fp4(x_scaled, self.grid)
        elif self.unbiased == "eden":
            x_fp4 = rtn_fp4(x_scaled, self.grid)
            
            x_fp4 = x_fp4.view(-1, self.hadamard_dim)
            x_scaled = x_scaled.view(-1, self.hadamard_dim)
            
            num = (x_scaled * x_scaled).sum(dim=-1, keepdim=True)
            denom = (x_scaled * x_fp4).sum(dim=-1, keepdim=True)
            correction = num / denom
            correction = torch.where(correction.isnan(), 1.0, correction)
            
            scales = self.apply_correction(scales, correction)
            
            x_fp4 = x_fp4.view(-1, self.group_dim)
        else:
            raise ValueError(f"Unsupported unbiased method: {self.unbiased}")

        return (x_had + (x_fp4 * scales * global_scale - x_had).detach()).reshape_as(x)

    def re_randomize(self):
        if self.rerotate == "signs":
            self.hadamard_matrix = self.hadamard_matrix @ torch.diag(
                torch.randint(
                    0, 2, (self.hadamard_dim,),
                    device=self.hadamard_matrix.device,
                    dtype=self.hadamard_matrix.dtype
                ) * 2 - 1
            )
        elif self.rerotate == "O32":
            gaussian_matrix = torch.randn(self.hadamard_dim, self.hadamard_dim, device=self.hadamard_matrix.device, dtype=self.hadamard_matrix.dtype)
            svd = torch.linalg.svd(gaussian_matrix)
            self.hadamard_matrix = svd[0] @ svd[2]
        elif self.rerotate is None:
            pass
        else:
            raise ValueError(f"Invalid rerotate value: {self.rerotate}")


class IsolatedEdenQuantizer(EdenSRQuantizer): # Specifically for testing backward without weight re-quant
    def forward(self, x):
        if (
            self.hadamard_dim == 1 and
            self.scale_dtype == "e4m3" and
            self.unbiased == "sr"
        ):
            return sr_1x16s_fp4_kernel_wrapper(
                x,
                self.scale_override,
                self.group_dim,
            )
        
        self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        self.grid = self.grid.to(x.device).to(x.dtype)
        
        x_had = (
            self.hadamard_matrix @ x.reshape(self.hadamard_dim, -1)
        ).reshape(-1, self.group_dim)
        scales = x_had.abs().max(dim=-1, keepdim=True)[0]
        
        scales, global_scale = self.round_scales(scales)
        
        x_scaled = x_had / scales / global_scale
        if self.unbiased == "no":
            x_fp4 = rtn_fp4(x_scaled, self.grid)
        elif self.unbiased == "sr":
            x_fp4 = sr_fp4(x_scaled, self.grid)
        elif self.unbiased == "eden":
            x_fp4 = rtn_fp4(x_scaled, self.grid)
            
            x_fp4 = x_fp4.view(-1, self.hadamard_dim)
            x_scaled = x_scaled.view(-1, self.hadamard_dim)
            
            num = (x_scaled * x_scaled).sum(dim=-1, keepdim=True)
            denom = (x_scaled * x_fp4).sum(dim=-1, keepdim=True)
            correction = num / denom
            correction = torch.where(correction.isnan(), 1.0, correction)
            
            scales = self.apply_correction(scales, correction)
            
            x_fp4 = x_fp4.view(-1, self.group_dim)
        else:
            raise ValueError(f"Unsupported unbiased method: {self.unbiased}")

        return (
            self.hadamard_matrix.T @ (x_fp4 * scales * global_scale).reshape(self.hadamard_dim, -1)
        ).reshape_as(x)
