from random import randint

from scipy.linalg import hadamard

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from src.models.quantization.quantizers.nvfp4_triton import rtn_1x16s_fp4_kernel_wrapper


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


def rerotate_hadamard(hadamard_matrix):
    signs = torch.diag(
        torch.randint(
            0, 2, (hadamard_matrix.size(0),),
            device=hadamard_matrix.device,
            dtype=hadamard_matrix.dtype
        ) * 2 - 1
    )
    return hadamard_matrix @ signs # NOTE: rerotate along last dim, inner dim for TN GEMM


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def eden_1x16s_fp4_kernel(
    x_ptr,
    hadamard_matrix_ptr,
    current_amax_ptr,
    output_ptr,
    next_amax_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
):    
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)
    
    # hadamard transform
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(hadamard_dim, hadamard_dim) 
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix) # not TN!, A @ B!
    
    # write amax for next iter
    tl.atomic_max(next_amax_ptr, tl.max(tl.abs(x_had)).to(tl.float32), sem="relaxed")
    
    # group
    x_grouped = tl.reshape(x_had, (BLOCK_SIZE // group_size, group_size))

    # amax
    scales_max = 255.99 # Not 448 because eden needs space to rescale up a bit sometimes after the correction
    val_max = 6.0 / scale_override
    amax = tl.load(current_amax_ptr)
    s_dec = tl.where(
        amax == 0.0,
        1.0,
        amax / scales_max / val_max,
    )
    
    # scale
    s_dec_b = tl.max(tl.abs(x_grouped), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(
        s_dec_b_e4m3 == 0,
        1.0,
        s_dec_b_e4m3,
    )
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    x_scaled_abs = tl.abs(x_scaled)
    x_scaled_sign = tl.where(
        x_scaled > 0,
        1,
        -1,
    )
    x_fp4 = tl.where(
        x_scaled_abs >= 5,
        6,
        tl.where(
            x_scaled_abs >= 3.5,
            4,
            tl.where(
                x_scaled_abs >= 2.5,
                3,
                tl.where(
                    x_scaled_abs >= 1.75,
                    2,
                    tl.where(
                        x_scaled_abs >= 1.25,
                        1.5,
                        tl.where(
                            x_scaled_abs >= 0.75,
                            1,
                            tl.where(
                                x_scaled_abs >= 0.25,
                                0.5,
                                0,
                            )
                        )
                    )
                )
            )
        )
    ) * x_scaled_sign
    
    # Calculate EDEN scale
    x_scaled = tl.reshape(x_scaled, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_fp4 = tl.reshape(x_fp4, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    
    num = tl.sum(x_scaled * x_scaled, axis=-1, keep_dims=True)
    denom = tl.sum(x_scaled * x_fp4, axis=-1, keep_dims=True)
    
    correction = tl.where(
        denom == 0.0,
        1.0,
        num / denom,
    )
    
    # Apply EDEN scale
    scales = tl.reshape(s_dec_b_e4m3, (BLOCK_SIZE // hadamard_dim, hadamard_dim // group_size))
    corrected_scales = tl.reshape(scales * correction, (BLOCK_SIZE // group_size, 1))
    
    bitscales = tl.cast(corrected_scales.to(tl.float8e4nv), tl.uint8, bitcast=True)
    prevscale = tl.cast((bitscales - 1), tl.float8e4nv, bitcast=True).to(tl.float32)
    currscale = tl.cast((bitscales), tl.float8e4nv, bitcast=True).to(tl.float32)
    nextscale = tl.cast((bitscales + 1), tl.float8e4nv, bitcast=True).to(tl.float32)
    
    up = tl.where(
        currscale > corrected_scales,
        currscale,
        nextscale,
    )
    down = tl.where(
        currscale > corrected_scales,
        prevscale,
        currscale,
    )
    
    prob_up = (corrected_scales - down) / (up - down)
    
    scale_start_idx = pid * (BLOCK_SIZE // group_size)
    scale_offsets = scale_start_idx + tl.arange(0, BLOCK_SIZE // group_size)
    sampled_prob = tl.rand(seed, scale_offsets).reshape(BLOCK_SIZE // group_size, 1)
    
    scales = tl.where(
        sampled_prob < prob_up,
        up,
        down,
    )
    scales = tl.reshape(scales, (BLOCK_SIZE // group_size, 1))
    x_fp4 = tl.reshape(x_fp4, (BLOCK_SIZE // group_size, group_size))
    
    # Reshape back to flat form for storage
    x_dequantized = x_fp4 * scales * s_dec
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    
    # store
    tl.store(output_ptr + offsets, x_dequantized_flat.to(x_ptr.dtype.element_ty), mask=mask)

@torch.compiler.disable()
def eden_1x16s_fp4_kernel_wrapper(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    scale_override: float,
    group_size: int,
    current_amax: torch.Tensor,
) -> [torch.Tensor, torch.Tensor]:
    hadamard_dim = hadamard_matrix.size(0)
    assert hadamard_matrix.size(1) == hadamard_dim
    assert x.numel() % hadamard_dim == 0
    assert hadamard_dim % group_size == 0
    
    x = x.contiguous()
    hadamard_matrix = hadamard_matrix.T.contiguous() # .T.contiguous() + tl.dot -> TN
    output = torch.empty_like(x)
    seed = randint(0, 1000000)
    
    next_amax = torch.zeros_like(current_amax)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    eden_1x16s_fp4_kernel[grid](
        x_ptr=x,
        hadamard_matrix_ptr=hadamard_matrix,
        current_amax_ptr=current_amax,
        output_ptr=output,
        next_amax_ptr=next_amax,
        n_elements=n_elements,
        hadamard_dim=hadamard_dim,
        scale_override=scale_override,
        group_size=group_size,
        seed=seed,
    )
    return output, next_amax


class AmaxStorage:
    def __init__(self):
        self.e_ht_amax = None
        self.weght_tht_amax = None
        self.e_tht_amax = None
        self.input_tht_amax = None
        
    def __repr__(self) -> str:
        fields = [
            ("e_ht_amax", self.e_ht_amax), 
            ("weght_tht_amax", self.weght_tht_amax), 
            ("e_tht_amax", self.e_tht_amax), 
            ("input_tht_amax", self.input_tht_amax)
        ]
        field_strs = []
        for name, val in fields:
            if val is not None:
                try:
                    v = val.item()
                except Exception:
                    v = val
                field_strs.append(f"{name}: {v:.3e}")
            else:
                field_strs.append(f"{name}: None")
        return "<AmaxStorage " + ", ".join(field_strs) + ">"
        

class Quartet_II_fn(torch.autograd.Function):
    group_size = 16
    forward_scale_override = 1.0
    backward_scale_override = (17 / 16) * 0.93
    hadamard_matrix = None

    @torch.compile(dynamic=False)
    @staticmethod
    def forward(ctx, input, weight, amax_storage: AmaxStorage, delayed_amax: bool, disable_forward_quant: bool, disable_backward_quant: bool):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.delayed_amax = delayed_amax
        ctx.amax_storage = amax_storage
        ctx.disable_backward_quant = disable_backward_quant
        
        if disable_forward_quant:
            input_fp4 = input
            weight_fp4 = weight
        else:
            input_fp4 = rtn_1x16s_fp4_kernel_wrapper(input, scale_override=Quartet_II_fn.forward_scale_override, group_size=Quartet_II_fn.group_size)
            weight_fp4 = rtn_1x16s_fp4_kernel_wrapper(weight, scale_override=Quartet_II_fn.forward_scale_override, group_size=Quartet_II_fn.group_size)

        ctx.save_for_backward(input_fp4, weight_fp4)
        return F.linear(input_fp4, weight_fp4)

    @torch.compile(dynamic=False)
    @staticmethod
    def backward(ctx, grad_output):
        # Load ctx and reshape
        input_fp4, weight_fp4 = ctx.saved_tensors
        
        input_fp4 = input_fp4.reshape(ctx.batch * ctx.seq, ctx.in_dim)
        grad_output = grad_output.reshape(ctx.batch * ctx.seq, ctx.out_dim)
        
        # Re-randomize the rotation
        Quartet_II_fn.hadamard_matrix = rerotate_hadamard(Quartet_II_fn.hadamard_matrix)
        
        # No backward quant if flag
        if ctx.disable_backward_quant:
            grad_input = F.linear(
                grad_output,
                weight_fp4.T,
                None,
            ).view(ctx.batch, ctx.seq, ctx.in_dim)
            
            grad_weight = F.linear(
                grad_output.T,
                input_fp4.T,
                None,
            )
            return grad_input, grad_weight, None, None, None, None
        
        # EW
        if ctx.amax_storage.e_ht_amax is None or not ctx.delayed_amax:
            ctx.amax_storage.e_ht_amax = (grad_output.reshape(-1, Quartet_II_fn.hadamard_matrix.size(0)) @ Quartet_II_fn.hadamard_matrix.T).amax().float()
        Quartet_II_fn.hadamard_matrix = Quartet_II_fn.hadamard_matrix.to(grad_output.dtype)
        e_ht_fp4, ctx.amax_storage.e_ht_amax = eden_1x16s_fp4_kernel_wrapper(grad_output, Quartet_II_fn.hadamard_matrix, Quartet_II_fn.backward_scale_override, 16, ctx.amax_storage.e_ht_amax)
        
        if ctx.amax_storage.weght_tht_amax is None or not ctx.delayed_amax:
            ctx.amax_storage.weght_tht_amax = (weight_fp4.T.reshape(-1, Quartet_II_fn.hadamard_matrix.size(0)) @ Quartet_II_fn.hadamard_matrix.T).amax().float()
        Quartet_II_fn.hadamard_matrix = Quartet_II_fn.hadamard_matrix.to(weight_fp4.dtype)
        weight_tht_fp4, ctx.amax_storage.weght_tht_amax = eden_1x16s_fp4_kernel_wrapper(weight_fp4.T, Quartet_II_fn.hadamard_matrix, Quartet_II_fn.backward_scale_override, 16, ctx.amax_storage.weght_tht_amax)
        
        grad_input = F.linear(
            e_ht_fp4,
            weight_tht_fp4,
            None,
        ).view(ctx.batch, ctx.seq, ctx.in_dim)

        # EtX
        if ctx.amax_storage.e_tht_amax is None or not ctx.delayed_amax:
            ctx.amax_storage.e_tht_amax = (grad_output.T.reshape(-1, Quartet_II_fn.hadamard_matrix.size(0)) @ Quartet_II_fn.hadamard_matrix.T).amax().float()
        Quartet_II_fn.hadamard_matrix = Quartet_II_fn.hadamard_matrix.to(grad_output.dtype)
        e_tht_fp4, ctx.amax_storage.e_tht_amax = eden_1x16s_fp4_kernel_wrapper(grad_output.T, Quartet_II_fn.hadamard_matrix, Quartet_II_fn.backward_scale_override, Quartet_II_fn.group_size, ctx.amax_storage.e_tht_amax)
        
        if ctx.amax_storage.input_tht_amax is None or not ctx.delayed_amax:
            ctx.amax_storage.input_tht_amax = (input_fp4.T.reshape(-1, Quartet_II_fn.hadamard_matrix.size(0)) @ Quartet_II_fn.hadamard_matrix.T).amax().float()
        Quartet_II_fn.hadamard_matrix = Quartet_II_fn.hadamard_matrix.to(input_fp4.dtype)
        input_tht_fp4, ctx.amax_storage.input_tht_amax = eden_1x16s_fp4_kernel_wrapper(input_fp4.T, Quartet_II_fn.hadamard_matrix, Quartet_II_fn.backward_scale_override, Quartet_II_fn.group_size, ctx.amax_storage.input_tht_amax)
        
        grad_weight = F.linear(
            e_tht_fp4,
            input_tht_fp4,
            None,
        )
        
        return grad_input, grad_weight, None, None, None, None


class Quartet_II_Linear(torch.nn.Linear):
    def __init__(self, *args, hadamard_dim=32, delayed_amax=False, disable_forward_quant=False, disable_backward_quant=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hadamard_dim = hadamard_dim
        self.delayed_amax = delayed_amax
        self.disable_forward_quant = disable_forward_quant
        self.disable_backward_quant = disable_backward_quant
        self.amax_storage = AmaxStorage()
        
        if Quartet_II_fn.hadamard_matrix is None:
            Quartet_II_fn.hadamard_matrix = get_hadamard_matrix(self.hadamard_dim, device="cuda", dtype=torch.float32)
        
    
    def forward(self, x):
        return Quartet_II_fn.apply(x, self.weight, self.amax_storage, self.delayed_amax, self.disable_forward_quant, self.disable_backward_quant)
