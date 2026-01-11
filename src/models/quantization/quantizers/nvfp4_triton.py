from random import randint

import torch
import triton
import triton.language as tl


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
    output_ptr,
    n_elements: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):        
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))
    
    # scale
    scales = tl.max(tl.abs(x_grouped), axis=-1, keep_dims=True)
    global_scale = tl.max(scales) / 447.99
    global_scale = tl.where(
        global_scale == 0,
        1.0,
        global_scale,
    )
    scales = scales / global_scale
    scales = scales.to(tl.float8e4nv).to(tl.float32)
    scales = tl.where(
        scales == 0,
        1.0,
        scales,
    )
    global_scale = global_scale / 6.0 * scale_override
    
    x_scaled = x_grouped / scales / global_scale
    
    # quantize
    x_scaled_abs = tl.abs(x_scaled)
    x_scaled_sign = tl.where(
        x_scaled > 0,
        1,
        -1,
    )
    x_fp4_abs = tl.where(
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
                            x_scaled_abs > 0.75,
                            1,
                            tl.where(
                                x_scaled_abs > 0.25,
                                0.5,
                                0.0,
                            )
                        )
                    )
                )
            )
        )
    )
    x_fp4 = x_fp4_abs * x_scaled_sign

    # dequantize
    x_dequantized = x_fp4 * scales * global_scale
    
    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    
    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def rtn_1x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    rtn_1x16s_fp4_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
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
def sr_1x16s_fp4_kernel(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
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
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))
    
    # scale
    scales = tl.max(tl.abs(x_grouped), axis=-1, keep_dims=True)
    global_scale = tl.max(scales) / 256.0
    global_scale = tl.where(
        global_scale == 0,
        1.0,
        global_scale,
    )
    scales = scales / global_scale
    scales = scales.to(tl.float8e4nv).to(tl.float32)
    scales = tl.where(
        scales == 0,
        1.0,
        scales,
    )
    global_scale = global_scale / 6.0 * (17 / 16) * scale_override
    
    x_scaled = x_grouped / scales / global_scale
    
    # quantize
    x_scaled_abs = tl.abs(x_scaled)
    x_scaled_sign = tl.where(
        x_scaled > 0,
        1,
        -1,
    )
    x_fp4_high = tl.where(
        x_scaled_abs > 4,
        6,
        tl.where(
            x_scaled_abs > 3,
            4,
            tl.where(
                x_scaled_abs > 2,
                3,
                tl.where(
                    x_scaled_abs > 1.5,
                    2,
                    tl.where(
                        x_scaled_abs > 1.0,
                        1.5,
                        tl.where(
                            x_scaled_abs > 0.5,
                            1,
                            0.5,
                        )
                    )
                )
            )
        )
    )
    
    x_fp4_low = tl.where(
        x_scaled_abs > 4,
        4,
        tl.where(
            x_scaled_abs > 3,
            3,
            tl.where(
                x_scaled_abs > 2,
                2,
                tl.where(
                    x_scaled_abs > 1.5,
                    1.5,
                    tl.where(
                        x_scaled_abs > 1.0,
                        1.0,
                        tl.where(
                            x_scaled_abs > 0.5,
                            0.5,
                            0.0,
                        )
                    )
                )
            )
        )
    )
    
    prob_up = (x_scaled_abs - x_fp4_low) / (x_fp4_high - x_fp4_low)
    sampled_prob = tl.rand(seed, offsets).reshape(BLOCK_SIZE // group_size, group_size)
    x_fp4 = tl.where(
        sampled_prob < prob_up,
        x_fp4_high,
        x_fp4_low,
    ) * x_scaled_sign

    # dequantize
    x_dequantized = x_fp4 * scales * global_scale
    
    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    
    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def sr_1x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    seed = randint(0, 1000000)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    sr_1x16s_fp4_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
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
    hadamard_matrix_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
):
    # load hadamard_matrix
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(hadamard_dim, hadamard_dim)
    
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)
    
    # hadamard transform
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)
    
    # group
    x_grouped = tl.reshape(x_had, (BLOCK_SIZE // group_size, group_size))
    
    # scale
    scales = tl.max(tl.abs(x_grouped), axis=-1, keep_dims=True)
    global_scale = tl.max(scales) / 256.0
    global_scale = tl.where(
        global_scale == 0,
        1.0,
        global_scale,
    )
    scales = scales / global_scale
    scales = scales.to(tl.float8e4nv).to(tl.float32)
    scales = tl.where(
        scales == 0,
        1.0,
        scales,
    )
    global_scale = global_scale / 6.0 * (17 / 16) * scale_override
    
    x_scaled = x_grouped / scales / global_scale
    
    # quantize
    x_scaled_abs = tl.abs(x_scaled)
    x_scaled_sign = tl.where(
        x_scaled > 0,
        1,
        -1,
    )
    x_fp4 = tl.where(
        x_scaled_abs > 5,
        6,
        tl.where(
            x_scaled_abs > 3.5,
            4,
            tl.where(
                x_scaled_abs > 2.5,
                3,
                tl.where(
                    x_scaled_abs > 1.75,
                    2,
                    tl.where(
                        x_scaled_abs > 1.25,
                        1.5,
                        tl.where(
                            x_scaled_abs > 0.75,
                            1,
                            tl.where(
                                x_scaled_abs > 0.25,
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
    scales = tl.reshape(scales, (BLOCK_SIZE // hadamard_dim, hadamard_dim // group_size))
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
    x_dequantized = x_fp4 * scales * global_scale
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    
    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def eden_1x16s_fp4_kernel_wrapper(
    x,
    hadamard_matrix,
    scale_override: float,
    group_size: int,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    seed = randint(0, 1000000)
    
    hadamard_dim = hadamard_matrix.size(0)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    eden_1x16s_fp4_kernel[grid](
        x_ptr=x,
        hadamard_matrix_ptr=hadamard_matrix,
        output_ptr=output,
        n_elements=n_elements,
        hadamard_dim=hadamard_dim,
        scale_override=scale_override,
        group_size=group_size,
        seed=seed,
    )
    return output
