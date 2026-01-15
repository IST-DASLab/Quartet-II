from random import randint

from scipy.linalg import hadamard

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from models.quantization.quantizers.nvfp4_triton import rtn_1x16s_fp4_kernel_wrapper, rtn_16x16s_fp4_kernel_wrapper, sr_1x16s_fp4_kernel_wrapper
from .quartet_2 import get_hadamard_matrix, rerotate_hadamard

class Nvidia_fn(torch.autograd.Function):
    group_size = 16
    hadamard_matrix = None

    @torch.compile(dynamic=False)
    @staticmethod
    def forward(ctx, input, weight, disable_forward_quant: bool, disable_backward_quant: bool, four_over_six: bool):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.disable_backward_quant = disable_backward_quant
        
        if disable_forward_quant:
            input_fp4 = input
            weight_fp4 = weight
        else:
            input_fp4 = rtn_1x16s_fp4_kernel_wrapper(input, scale_override=1.0, group_size=Nvidia_fn.group_size, four_over_six=four_over_six)
            weight_fp4 = rtn_16x16s_fp4_kernel_wrapper(weight, scale_override=1.0, group_size=Nvidia_fn.group_size, four_over_six=four_over_six)

        ctx.save_for_backward(input, weight_fp4)
        return F.linear(input_fp4, weight_fp4)

    @torch.compile(dynamic=False)
    @staticmethod
    def backward(ctx, grad_output):
        # Load ctx and reshape
        input, weight_fp4 = ctx.saved_tensors
        
        input = input.reshape(ctx.batch * ctx.seq, ctx.in_dim)
        grad_output = grad_output.reshape(ctx.batch * ctx.seq, ctx.out_dim)
        
        # Re-randomize the rotation
        Nvidia_fn.hadamard_matrix = rerotate_hadamard(Nvidia_fn.hadamard_matrix)
        
        # No backward quant if flag
        if ctx.disable_backward_quant:
            grad_input = F.linear(
                grad_output,
                weight_fp4.T,
                None,
            ).view(ctx.batch, ctx.seq, ctx.in_dim)
            
            grad_weight = F.linear(
                grad_output.T,
                input.T,
                None,
            )
            return grad_input, grad_weight, None, None, None
        
        # EW
        e_fp4 = sr_1x16s_fp4_kernel_wrapper(grad_output, (17 / 16), Nvidia_fn.group_size, False)
        
        grad_input = F.linear(
            e_fp4,
            weight_fp4.T,
            None,
        ).view(ctx.batch, ctx.seq, ctx.in_dim)

        # EtX
        e_tht = (grad_output.T.reshape(-1, Nvidia_fn.hadamard_matrix.size(0)) @ Nvidia_fn.hadamard_matrix.T).reshape(ctx.out_dim, ctx.batch * ctx.seq)
        e_tht_fp4 = sr_1x16s_fp4_kernel_wrapper(e_tht, (17 / 16), Nvidia_fn.group_size, False)
        
        input_tht = (input.T.reshape(-1, Nvidia_fn.hadamard_matrix.size(0)) @ Nvidia_fn.hadamard_matrix.T).reshape(ctx.in_dim, ctx.batch * ctx.seq)
        input_tht_fp4 = rtn_1x16s_fp4_kernel_wrapper(input_tht, 1.0, Nvidia_fn.group_size, False)
        
        grad_weight = F.linear(
            e_tht_fp4,
            input_tht_fp4,
            None,
        )
        
        return grad_input, grad_weight, None, None, None


class NvidiaLinear(torch.nn.Linear):
    def __init__(self, *args, hadamard_dim=16, disable_forward_quant=False, disable_backward_quant=False, four_over_six=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hadamard_dim = hadamard_dim
        self.disable_forward_quant = disable_forward_quant
        self.disable_backward_quant = disable_backward_quant
        self.four_over_six = four_over_six
        
        if Nvidia_fn.hadamard_matrix is None:
            Nvidia_fn.hadamard_matrix = get_hadamard_matrix(self.hadamard_dim, device="cuda", dtype=torch.float32)
        
    
    def forward(self, x):
        return Nvidia_fn.apply(x, self.weight, self.disable_forward_quant, self.disable_backward_quant, self.four_over_six)
