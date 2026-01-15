from random import randint

import torch
import triton
import triton.language as tl


@triton.jit
def rtn_fp4(x):
    x_abs = tl.abs(x)
    x_sign = tl.where(
        x > 0,
        1,
        -1,
    )
    x_fp4_abs = tl.where(
        x_abs >= 5,
        6,
        tl.where(
            x_abs >= 3.5,
            4,
            tl.where(
                x_abs >= 2.5,
                3,
                tl.where(
                    x_abs >= 1.75,
                    2,
                    tl.where(
                        x_abs >= 1.25,
                        1.5,
                        tl.where(
                            x_abs >= 0.75,
                            1,
                            tl.where(
                                x_abs >= 0.25,
                                0.5,
                                0.0,
                            )
                        )
                    )
                )
            )
        )
    )
    return x_fp4_abs * x_sign


@triton.jit
def sr_fp4(x, sampled_prob):
    x_abs = tl.abs(x)
    x_sign = tl.where(
        x > 0,
        1,
        -1,
    )
    x_fp4_high = tl.where(
        x_abs >= 4,
        6,
        tl.where(
            x_abs >= 3,
            4,
            tl.where(
                x_abs >= 2,
                3,
                tl.where(
                    x_abs >= 1.5,
                    2,
                    tl.where(
                        x_abs >= 1.0,
                        1.5,
                        tl.where(
                            x_abs >= 0.5,
                            1,
                            0.5,
                        )
                    )
                )
            )
        )
    )
    
    x_fp4_low = tl.where(
        x_abs > 4,
        4,
        tl.where(
            x_abs > 3,
            3,
            tl.where(
                x_abs > 2,
                2,
                tl.where(
                    x_abs > 1.5,
                    1.5,
                    tl.where(
                        x_abs > 1.0,
                        1.0,
                        tl.where(
                            x_abs > 0.5,
                            0.5,
                            0.0,
                        )
                    )
                )
            )
        )
    )
    
    prob_up = (x_abs - x_fp4_low) / (x_fp4_high - x_fp4_low)
    return tl.where(
        sampled_prob < prob_up,
        x_fp4_high,
        x_fp4_low,
    ) * x_sign


@triton.jit
def get_scales(x, amax, val_max, scales_max):
    s_dec = tl.where(
        amax == 0.0,
        1.0,
        amax / scales_max / val_max,
    )
    
    s_dec_b = tl.max(tl.abs(x), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(
        s_dec_b_e4m3 == 0,
        1.0,
        s_dec_b_e4m3,
    )
    return s_dec_b_e4m3, s_dec


@triton.jit
def get_alt_scales(x, val_max, s_dec):    
    s_dec_b = tl.max(tl.abs(x), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b * (6/4) / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(
        s_dec_b_e4m3 == 0,
        1.0,
        s_dec_b_e4m3,
    )
    return s_dec_b_e4m3


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
def rtn_1x16s_fp4_kernel(
    x_ptr,
    amax_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    four_over_six: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):        
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))
    
    # amax
    scales_max = 256.00 if four_over_six else 448.00
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)
    
    s_dec_b_e4m3, s_dec = get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    x_fp4 = rtn_fp4(x_scaled)
    x_dequantized = x_fp4 * (s_dec_b_e4m3 * s_dec)
    
    if not four_over_six:
        best_x_dequantized = x_dequantized
    else:
        alt_s_dec_b_e4m3 = get_alt_scales(x_grouped, val_max, s_dec)
        alt_x_scaled = x_grouped / (alt_s_dec_b_e4m3 * s_dec)
        
        alt_x_fp4 = rtn_fp4(alt_x_scaled)
        alt_x_dequantized = alt_x_fp4 * (alt_s_dec_b_e4m3 * s_dec)
        
        error_six = tl.sum((x_grouped - x_dequantized) * (x_grouped - x_dequantized), axis=-1, keep_dims=True)
        error_four = tl.sum((x_grouped - alt_x_dequantized) * (x_grouped - alt_x_dequantized), axis=-1, keep_dims=True)
        
        best_x_dequantized = tl.where(
            error_six <= error_four,
            x_dequantized,
            alt_x_dequantized,
        )
    
    x_dequantized_flat = tl.reshape(best_x_dequantized, (BLOCK_SIZE,))
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def rtn_1x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
    four_over_six: bool,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    rtn_1x16s_fp4_kernel[grid](
        x_ptr=x,
        amax_ptr=x.abs().max(),
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
        four_over_six=four_over_six,
    )
    return output

class rtn_1x16s_fp4_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_override, group_size, four_over_six):
        return rtn_1x16s_fp4_kernel_wrapper(x, scale_override, group_size, four_over_six)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
    ],
    key=[],
)
@triton.jit
def rtn_16x16s_fp4_kernel(
    x_ptr,
    amax_ptr,
    output_ptr,
    n_row: tl.constexpr,
    n_col: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    four_over_six: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):        
    # load x
    pidx = tl.program_id(0)
    pidy = tl.program_id(1)
    start_idx = pidx * BLOCK_SIZE
    start_idy = pidy * BLOCK_SIZE
    offsets = (
        (start_idx + tl.arange(0, BLOCK_SIZE))[:, None] * n_col +
        (start_idy + tl.arange(0, BLOCK_SIZE))[None, :]
    )
    mask = offsets < n_row * n_col
    x_flat = tl.load(x_ptr + offsets, mask=mask)
    # [BLOCK_SIZE, BLOCK_SIZE]
    
    # group
    x_grouped =  x_flat.reshape(
        BLOCK_SIZE // group_size, group_size, BLOCK_SIZE // group_size, group_size
    ).permute(
        (0, 2, 1, 3),
    ).reshape(
        (BLOCK_SIZE // group_size * BLOCK_SIZE // group_size, group_size * group_size),
    )
    
    # scale
    scales_max = 256.00 if four_over_six else 447.99
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)
    
    s_dec_b_e4m3, s_dec = get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    x_fp4 = rtn_fp4(x_scaled)
    x_dequantized = x_fp4 * (s_dec_b_e4m3 * s_dec)
    # [BLOCK_SIZE // group_size * BLOCK_SIZE // group_size, group_size * group_size]
    
    if not four_over_six:
        best_x_dequantized = x_dequantized
    else:
        alt_s_dec_b_e4m3 = get_alt_scales(x_grouped, val_max, s_dec)
        alt_x_scaled = x_grouped / (alt_s_dec_b_e4m3 * s_dec)
        
        alt_x_fp4 = rtn_fp4(alt_x_scaled)
        alt_x_dequantized = alt_x_fp4 * (alt_s_dec_b_e4m3 * s_dec)
        
        error_six = tl.sum((x_grouped - x_dequantized) * (x_grouped - x_dequantized), axis=-1, keep_dims=True)
        error_four = tl.sum((x_grouped - alt_x_dequantized) * (x_grouped - alt_x_dequantized), axis=-1, keep_dims=True)
        
        best_x_dequantized = tl.where(
            error_six <= error_four,
            x_dequantized,
            alt_x_dequantized,
        )

    x_dequantized_flat = tl.reshape(
        tl.permute(
            tl.reshape(best_x_dequantized, (BLOCK_SIZE // group_size, BLOCK_SIZE // group_size, group_size, group_size)),
            (0, 2, 1, 3),
        ),
        (BLOCK_SIZE, BLOCK_SIZE),
    )
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def rtn_16x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
    four_over_six: bool,
):
    assert x.dim() == 2
    x = x.contiguous()
    output = torch.empty_like(x)
    
    n_row = x.size(0)
    n_col = x.size(1)
    grid = lambda meta: (triton.cdiv(n_row, meta["BLOCK_SIZE"]),triton.cdiv(n_col, meta["BLOCK_SIZE"]))
    
    rtn_16x16s_fp4_kernel[grid](
        x_ptr=x,
        amax_ptr=x.abs().max(),
        output_ptr=output,
        n_row=n_row,
        n_col=n_col,
        scale_override=scale_override,
        group_size=group_size,
        four_over_six=four_over_six,
    )
    return output

class rtn_16x16s_fp4_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_override, group_size, four_over_six):
        return rtn_16x16s_fp4_kernel_wrapper(x, scale_override, group_size, four_over_six)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


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
def sr_1x16s_fp4_kernel(
    x_ptr,
    amax_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    four_over_six: tl.constexpr,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
):
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))
    
    # amax
    scales_max = 256.00 if four_over_six else 448.00
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)
    
    s_dec_b_e4m3, s_dec = get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    sampled_prob = tl.rand(seed, offsets).reshape(BLOCK_SIZE // group_size, group_size)
    x_fp4 = sr_fp4(x_scaled, sampled_prob)
    x_dequantized = x_fp4 * (s_dec_b_e4m3 * s_dec)
    
    if not four_over_six:
        best_x_dequantized = x_dequantized
    else:
        alt_s_dec_b_e4m3 = get_alt_scales(x_grouped, val_max, s_dec)
        alt_x_scaled = x_grouped / (alt_s_dec_b_e4m3 * s_dec)
        
        alt_sampled_prob = tl.rand(seed + 1, offsets).reshape(BLOCK_SIZE // group_size, group_size)
        alt_x_fp4 = sr_fp4(alt_x_scaled, alt_sampled_prob)
        alt_x_dequantized = alt_x_fp4 * (alt_s_dec_b_e4m3 * s_dec)
        
        error_six = tl.sum((x_grouped - x_dequantized) * (x_grouped - x_dequantized), axis=-1, keep_dims=True)
        error_four = tl.sum((x_grouped - alt_x_dequantized) * (x_grouped - alt_x_dequantized), axis=-1, keep_dims=True)
        
        best_x_dequantized = tl.where(
            error_six <= error_four,
            x_dequantized,
            alt_x_dequantized,
        )

    x_dequantized_flat = tl.reshape(best_x_dequantized, (BLOCK_SIZE,))
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def sr_1x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
    four_over_six: bool,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    seed = randint(0, 1000000)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    sr_1x16s_fp4_kernel[grid](
        x_ptr=x,
        amax_ptr=x.abs().max(),
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
        four_over_six=four_over_six,
        seed=seed,
    )
    return output


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
    amax_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    four_over_six: tl.constexpr,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
):    
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))

    # scale
    scales_max = 255.99
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)
    
    s_dec_b_e4m3, s_dec = get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    x_fp4 = rtn_fp4(x_scaled)
    
    if not four_over_six:
        best_x_scaled = x_scaled
        best_x_fp4 = x_fp4
        best_s_dec_b_e4m3 = s_dec_b_e4m3
    else:
        alt_s_dec_b_e4m3 = get_alt_scales(x_grouped, val_max, s_dec)
        alt_x_scaled = x_grouped / (alt_s_dec_b_e4m3 * s_dec)
        
        alt_x_fp4 = rtn_fp4(alt_x_scaled)
        
        x_dequantized = x_fp4 * (s_dec_b_e4m3 * s_dec)
        alt_x_dequantized = alt_x_fp4 * (alt_s_dec_b_e4m3 * s_dec)
        
        error_six = tl.sum((x_grouped - x_dequantized) * (x_grouped - x_dequantized), axis=-1, keep_dims=True)
        error_four = tl.sum((x_grouped - alt_x_dequantized) * (x_grouped - alt_x_dequantized), axis=-1, keep_dims=True)
        
        best_x_scaled = tl.where(
            error_six <= error_four,
            x_scaled,
            alt_x_scaled,
        )
        
        best_x_fp4 = tl.where(
            error_six <= error_four,
            x_fp4,
            alt_x_fp4,
        )
        
        best_s_dec_b_e4m3 = tl.where(
            error_six <= error_four,
            s_dec_b_e4m3,
            alt_s_dec_b_e4m3,
        )
    
    # Calculate EDEN scale
    x_scaled = tl.reshape(best_x_scaled, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_fp4 = tl.reshape(best_x_fp4, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    
    num = tl.sum(x_scaled * x_scaled, axis=-1, keep_dims=True)
    denom = tl.sum(x_scaled * x_fp4, axis=-1, keep_dims=True)
    
    correction = tl.where(
        denom == 0.0,
        1.0,
        num / denom,
    )
    
    # Apply EDEN scale
    scales = tl.reshape(best_s_dec_b_e4m3, (BLOCK_SIZE // hadamard_dim, hadamard_dim // group_size))
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
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def eden_1x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    hadamard_dim: int,
    group_size: int,
    four_over_six: bool,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    seed = randint(0, 1000000)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    eden_1x16s_fp4_kernel[grid](
        x_ptr=x,
        amax_ptr=x.abs().max(),
        output_ptr=output,
        n_elements=n_elements,
        hadamard_dim=hadamard_dim,
        scale_override=scale_override,
        group_size=group_size,
        four_over_six=four_over_six,
        seed=seed,
    )
    return output
