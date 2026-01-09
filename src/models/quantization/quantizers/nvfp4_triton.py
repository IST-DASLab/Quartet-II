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
def isolated_eden_sr_kernel(
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
def isolated_eden_sr_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
):    
    # Make sure inputs are contiguous
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    

    seed = randint(0, 1000000)

    
    # Get total number of elements and calculate grid for launching the kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch optimized kernel
    isolated_eden_sr_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
        seed=seed,
    )
    
    return output
