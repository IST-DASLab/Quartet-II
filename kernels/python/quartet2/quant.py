from collections import namedtuple
from enum import Enum

import torch
from . import _quartet2


# torch-compile compatible wrapping
@torch.library.custom_op("quartet2::four_six", mutates_args=("o", "s", "t"))
def _four_six_fp4_op(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor, amax: torch.Tensor, scale_override: float) -> None:
    _quartet2.four_six_fp4(o, s.view(torch.uint8), t, x.detach(), amax, scale_override)


@torch.library.custom_op("quartet2::rtn", mutates_args=("o", "s", "t"))
def _rtn_fp4_op(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor, amax: torch.Tensor, scale_override: float) -> None:
    _quartet2.rtn_fp4(o, s.view(torch.uint8), t, x.detach(), amax, scale_override)


@torch.library.custom_op("quartet2::eden", mutates_args=("o", "s", "t"))
def _eden_fp4_op(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor, amax: torch.Tensor, scale_override: float, seed: torch.Tensor) -> None:
    _quartet2.eden_fp4(o, s.view(torch.uint8), t, x.detach(), amax, scale_override, seed)


@torch.library.custom_op("quartet2::group_transform_eden", mutates_args=("o", "s", "t", "scratch_scales", "scratch_max"))
def _group_transform_128_eden(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                              scratch_scales: torch.Tensor, scratch_max: torch.Tensor,
                              h: torch.Tensor, x: torch.Tensor, seed: torch.Tensor, fp4_max: float, fp8_max: float, transpose: bool) -> None:
    _quartet2.group_transform_128_eden(
        o,
        s.view(torch.uint8),
        t,
        scratch_scales,
        scratch_max,
        h,
        x.detach(),
        seed,
        fp4_max=fp4_max,
        fp8_max=fp8_max,
        transpose=transpose,
    )


@torch.library.custom_op("quartet2::rht128_eden", mutates_args=("o", "s", "t", "scratch_scales", "scratch_max"))
def _transform_rht128_eden(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                           scratch_scales: torch.Tensor, scratch_max: torch.Tensor,
                           h: torch.Tensor, x: torch.Tensor, seed: torch.Tensor, fp4_max: float, fp8_max: float, transpose: bool) -> None:
    _quartet2.rht128_eden(
        o,
        s.view(torch.uint8),
        t,
        scratch_scales,
        scratch_max,
        h,
        x.detach(),
        seed,
        fp4_max=fp4_max,
        fp8_max=fp8_max,
        transpose=transpose,
    )


@torch.library.custom_op("quartet2::dequant_tp_had_quant", mutates_args=("o", "s", "t", "scratch_scales", "scratch_max"))
def _dequant_tp_had_quant(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                          scratch_scales: torch.Tensor, scratch_max: torch.Tensor,
                          h: torch.Tensor, x: torch.Tensor, x_group_scales: torch.Tensor, x_tensor_scale: torch.Tensor,
                          seed: torch.Tensor, fp4_max: float, fp8_max: float) -> None:
    _quartet2.dequant_tp_had_quant(
        o,
        s,
        t,
        scratch_scales,
        scratch_max,
        h,
        x,
        x_group_scales,
        x_tensor_scale,
        seed,
        fp4_max=fp4_max,
        fp8_max=fp8_max,
    )


@torch.library.custom_op("quartet2::had128_requant", mutates_args=("o", "s", "t", "scratch_scales", "scratch_max"))
def _rht128_requant(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                    scratch_scales: torch.Tensor, scratch_max: torch.Tensor,
                    h: torch.Tensor, x: torch.Tensor, x_group_scales: torch.Tensor, x_tensor_scale: torch.Tensor,
                    seed: torch.Tensor, fp4_max: float, fp8_max: float) -> None:
    _quartet2.rht128_requant(
        o,
        s,
        t,
        scratch_scales,
        scratch_max,
        h,
        x,
        x_group_scales,
        x_tensor_scale,
        seed,
        fp4_max=fp4_max,
        fp8_max=fp8_max,
    )



NVFP4Quant = namedtuple("NVFP4Quant", ["fp4", "micro_scales", "tensor_scale"])


class NVFP4QuantMode(Enum):
    RNE = 0
    FOUR_SIX = 1
    EDEN = 2


def quant_fp4(x, *, scale_override: float, amax: torch.Tensor = None, mode=NVFP4QuantMode.RNE, seed: torch.Tensor=None) -> NVFP4Quant:
    q = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=torch.uint8)
    s = torch.empty((x.shape[0], x.shape[1] // 16), device=x.device, dtype=torch.float8_e4m3fn)
    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert x.shape[0] % 128 == 0
    assert x.shape[1] % 128 == 0

    if amax is None:
        amax = torch.max(torch.abs(x)).to(torch.float32)
    else:
        assert amax.dtype == torch.float32
    global_scale = torch.empty((), device=x.device, dtype=torch.float32)
    if mode == NVFP4QuantMode.FOUR_SIX:
        _four_six_fp4_op(q, s, global_scale, x, amax, scale_override)
    elif mode == NVFP4QuantMode.RNE:
        _rtn_fp4_op(q, s, global_scale, x, amax, scale_override)
    elif mode == NVFP4QuantMode.EDEN:
        assert seed is not None
        _eden_fp4_op(q, s, global_scale, x, amax, scale_override, seed)
    else:
        raise ValueError(f"Unknown NVFP4QuantMode: {mode}")

    return NVFP4Quant(q, s, global_scale)


def quant_had_eden(
        *,
        out: torch.Tensor = None,
        group_scales: torch.Tensor = None,
        tensor_scale: torch.Tensor = None,
        scratch_scales: torch.Tensor = None,
        scratch_amax: torch.Tensor = None,
        h: torch.Tensor,
        x: torch.Tensor,
        seed: torch.Tensor = None,
        scale_override: float = 1.0,
        transpose: bool = False):
    """
    Transforms tensor `x` using `h` as `x h^t`, and converts the result to nvfp4.

    :param out: Target tensor to place the fp4 values. Will be allocated if not provided
    :param group_scales: Target tensor to place the fp8 group scales. Will be allocated if not provided
    :param tensor_scale: Target scalar to place the fp32 tensor scale. Will be allocated if not provided
    :param scratch_scales: Scratch tensor to use for intermediate bf16 group scales. Will be allocated if not provided.
    :param scratch_amax: Scratch tensor to use for intermediate abs-max values (size 1). Will be allocated if not provided.
    :param h: 128x128 transform matrix
    :param x: input
    :param seed: seed to use for deterministic stochastic rounding.
    :param scale_override: Modulate the scaling factor to use when converting to fp4.
    :param transpose: transpose `x` before transforming
    :return: (out, group_scales, tensor_scale)
    """
    if seed is None:
        seed = torch.randint(0, 2**32, (), dtype=torch.int64)

    assert x.dim() == 2
    rows, cols = x.shape
    if transpose:
        rows, cols = cols, rows

    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert h.dtype == torch.bfloat16
    assert h.is_cuda
    assert h.is_contiguous()
    assert h.shape == (128, 128)

    if out is None:
        out = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x.device)
    else:
        out = out.reshape(-1, out.shape[-1])

    if group_scales is None:
        group_scales = torch.empty((rows, cols // 16), device=x.device, dtype=torch.float8_e4m3fn)
    else:
        group_scales = group_scales.reshape(-1, group_scales.shape[-1])

    if tensor_scale is None:
        tensor_scale = torch.empty((), device=x.device, dtype=torch.float32)

    if scratch_scales is None:
        scratch_scales = torch.empty((rows * cols // 16), device=x.device, dtype=torch.bfloat16)
    if scratch_amax is None:
        scratch_amax = torch.empty((), device=x.device, dtype=torch.uint32)

    _group_transform_128_eden(
        out,
        group_scales,
        tensor_scale,
        scratch_scales,
        scratch_amax,
        h,
        x,
        seed,
        fp4_max=6.0 / scale_override,
        fp8_max=256.0,
        transpose=transpose,
    )

    group_scales = group_scales.reshape(rows, cols // 16)

    return NVFP4Quant(out, group_scales, tensor_scale)



def dequant_tp_had_eden(
        *,
        out: torch.Tensor = None,
        group_scales: torch.Tensor = None,
        tensor_scale: torch.Tensor = None,
        scratch_scales: torch.Tensor = None,
        scratch_amax: torch.Tensor = None,
        h: torch.Tensor,
        x: torch.Tensor,
        x_group_scales: torch.Tensor,
        x_tensor_scale: torch.Tensor,
        seed: int = None,
        scale_override: float = 1.0,
):
    """
    Transforms tensor `x` using `h` as `x^t h^t`, and converts the result to nvfp4.
    `x` is given as a nvfp4-quantized tensor.

    Note: The output of this function is swizzled, in that the elements of each group of 16 are written in order
    `[0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]`. If the result is used for a matrix product with another
    value with the same swizzling order, the result will be correct. Otherwise, it is recommended to rearrange `h`
    such that the swizzling actually corresponds to the desired output.

    :param out: Target tensor to place the fp4 values. Will be allocated if not provided
    :param group_scales: Target tensor to place the fp8 group scales. Will be allocated if not provided
    :param tensor_scale: Target scalar to place the fp32 tensor scale. Will be allocated if not provided
    :param scratch_scales: Scratch tensor to use for intermediate bf16 group scales. Will be allocated if not provided.
    :param scratch_amax: Scratch tensor to use for intermediate abs-max values (size 1). Will be allocated if not provided.
    :param h: 128x128 transform matrix
    :param x: fp4 quants for input `x`
    :param x_group_scales: group scales for `x`
    :param x_tensor_scale: tensor scale for `x`
    :param seed: seed to use for deterministic stochastic rounding.
    :param scale_override: Modulate the scaling factor to use when converting to fp4.
    :return: (out, group_scales, tensor_scale)
    """
    if seed is None:
        seed = torch.randint(0, 2**32, (), dtype=torch.int64)

    assert x.dim() == 2
    rows, cols = x.shape
    rows, cols = 2*cols, rows  # transpose and take into account 2 vals per element

    assert x.dtype == torch.uint8
    assert x.is_cuda
    assert x.is_contiguous()
    assert x_group_scales.dtype == torch.float8_e4m3fn
    assert x_group_scales.shape[0] == x.shape[0]
    assert x_group_scales.shape[1] == x.shape[1] // 8
    assert x_group_scales.is_cuda
    assert x_group_scales.is_contiguous()
    assert x_tensor_scale.dim() == 0
    assert x_tensor_scale.dtype == torch.float32

    assert h.dtype == torch.bfloat16
    assert h.is_cuda
    assert h.is_contiguous()
    assert h.shape == (128, 128)

    if out is None:
        out = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x.device)
    else:
        out = out.reshape(-1, out.shape[-1])

    if group_scales is None:
        group_scales = torch.empty((rows, cols // 16), device=x.device, dtype=torch.float8_e4m3fn)
    else:
        group_scales = group_scales.reshape(-1, group_scales.shape[-1])

    if tensor_scale is None:
        tensor_scale = torch.empty((), device=x.device, dtype=torch.float32)

    if scratch_scales is None:
        scratch_scales = torch.empty((rows * cols // 16), device=x.device, dtype=torch.bfloat16)

    if scratch_amax is None:
        scratch_amax = torch.empty((), device=x.device, dtype=torch.uint32)

    _dequant_tp_had_quant(
        out,
        group_scales.view(torch.uint8),
        tensor_scale,
        scratch_scales,
        scratch_amax,
        h,
        x,
        x_group_scales.view(torch.uint8),
        x_tensor_scale,
        seed,
        fp4_max=6.0 / scale_override,
        fp8_max=256.0,
    )

    group_scales = group_scales.reshape(rows, cols // 16)

    return NVFP4Quant(out, group_scales, tensor_scale)


def rht128_quant_eden(
        *,
        out: torch.Tensor = None,
        group_scales: torch.Tensor = None,
        tensor_scale: torch.Tensor = None,
        scratch_scales: torch.Tensor = None,
        scratch_amax: torch.Tensor = None,
        h: torch.Tensor,
        x: torch.Tensor,
        seed: torch.Tensor = None,
        scale_override: float = 1.0,
        transpose: bool = False):
    """
    Transforms tensor `x` using `h` as `x h^t`, and converts the result to nvfp4.

    :param out: Target tensor to place the fp4 values. Will be allocated if not provided
    :param group_scales: Target tensor to place the fp8 group scales. Will be allocated if not provided
    :param tensor_scale: Target scalar to place the fp32 tensor scale. Will be allocated if not provided
    :param scratch_scales: Scratch tensor to use for intermediate bf16 group scales. Will be allocated if not provided.
    :param scratch_amax: Scratch tensor to use for intermediate abs-max values (size 1). Will be allocated if not provided.
    :param h: The top 16x128 slice of a 128x128 hadamard matrix. The rest of the matrix is generated from this through
        Sylvester's construction, except that rows are reordered as implemented in :func:`quartet2.rht.swizzle_hadamard`.
    :param x: input
    :param seed: seed to use for deterministic stochastic rounding.
    :param scale_override: Modulate the scaling factor to use when converting to fp4.
    :param transpose: transpose `x` before transforming
    :return: (out, group_scales, tensor_scale)
    """
    if seed is None:
        seed = torch.randint(0, 2**32, (), dtype=torch.int64)

    assert x.dim() == 2
    rows, cols = x.shape
    if transpose:
        rows, cols = cols, rows

    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert h.dtype == torch.bfloat16
    assert h.is_cuda
    assert h.is_contiguous()

    if out is None:
        out = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x.device)
    else:
        out = out.reshape(-1, out.shape[-1])

    if group_scales is None:
        group_scales = torch.empty((rows, cols // 16), device=x.device, dtype=torch.float8_e4m3fn)
    else:
        group_scales = group_scales.reshape(-1, group_scales.shape[-1])

    if tensor_scale is None:
        tensor_scale = torch.empty((), device=x.device, dtype=torch.float32)

    if scratch_scales is None:
        scratch_scales = torch.empty((rows * cols // 16), device=x.device, dtype=torch.bfloat16)
    if scratch_amax is None:
        scratch_amax = torch.empty((), device=x.device, dtype=torch.uint32)

    _transform_rht128_eden(
        out,
        group_scales,
        tensor_scale,
        scratch_scales,
        scratch_amax,
        h,
        x,
        seed,
        fp4_max=6.0 / scale_override,
        fp8_max=256.0,
        transpose=transpose,
    )

    group_scales = group_scales.reshape(rows, cols // 16)

    return NVFP4Quant(out, group_scales, tensor_scale)




def rht128_requant(
        *,
        out: torch.Tensor = None,
        group_scales: torch.Tensor = None,
        tensor_scale: torch.Tensor = None,
        scratch_scales: torch.Tensor = None,
        scratch_amax: torch.Tensor = None,
        h: torch.Tensor,
        x: torch.Tensor,
        x_group_scales: torch.Tensor,
        x_tensor_scale: torch.Tensor,
        seed: torch.Tensor = None,
        scale_override: float = 1.0,
):
    """
    Transforms tensor `x` using `h` as `x^t h^t`, and converts the result to nvfp4.
    `x` is given as a nvfp4-quantized tensor.

    :param out: Target tensor to place the fp4 values. Will be allocated if not provided
    :param group_scales: Target tensor to place the fp8 group scales. Will be allocated if not provided
    :param tensor_scale: Target scalar to place the fp32 tensor scale. Will be allocated if not provided
    :param scratch_scales: Scratch tensor to use for intermediate bf16 group scales. Will be allocated if not provided.
    :param scratch_amax: Scratch tensor to use for intermediate abs-max values (size 1). Will be allocated if not provided.
    :param h: The top 16x128 slice of a 128x128 hadamard matrix. The rest of the matrix is generated from this through
        Sylvester's construction, except that rows are reordered as implemented in :func:`quartet2.rht.swizzle_hadamard`.
    :param x: fp4 quants for input `x`
    :param x_group_scales: group scales for `x`
    :param x_tensor_scale: tensor scale for `x`
    :param seed: seed to use for deterministic stochastic rounding.
    :param scale_override: Modulate the scaling factor to use when converting to fp4.
    :return: (out, group_scales, tensor_scale)
    """
    if seed is None:
        seed = torch.randint(0, 2**32, (), dtype=torch.int64)

    assert x.dim() == 2
    rows, cols = x.shape
    rows, cols = 2*cols, rows  # transpose and take into account 2 vals per element

    assert x.dtype == torch.uint8
    assert x.is_cuda
    assert x.is_contiguous()
    assert x_group_scales.dtype == torch.float8_e4m3fn
    assert x_group_scales.shape[0] == x.shape[0]
    assert x_group_scales.shape[1] == x.shape[1] // 8
    assert x_group_scales.is_cuda
    assert x_group_scales.is_contiguous()
    assert x_tensor_scale.dim() == 0
    assert x_tensor_scale.dtype == torch.float32

    assert h.dtype == torch.bfloat16
    assert h.is_cuda
    assert h.is_contiguous()

    if out is None:
        out = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x.device)
    else:
        out = out.reshape(-1, out.shape[-1])

    if group_scales is None:
        group_scales = torch.empty((rows, cols // 16), device=x.device, dtype=torch.float8_e4m3fn)
    else:
        group_scales = group_scales.reshape(-1, group_scales.shape[-1])

    if tensor_scale is None:
        tensor_scale = torch.empty((), device=x.device, dtype=torch.float32)

    if scratch_scales is None:
        scratch_scales = torch.empty((rows * cols // 16), device=x.device, dtype=torch.bfloat16)

    if scratch_amax is None:
        scratch_amax = torch.empty((), device=x.device, dtype=torch.uint32)

    _rht128_requant(
        out,
        group_scales.view(torch.uint8),
        tensor_scale,
        scratch_scales,
        scratch_amax,
        h,
        x,
        x_group_scales.view(torch.uint8),
        x_tensor_scale,
        seed,
        fp4_max=6.0 / scale_override,
        fp8_max=256.0,
    )

    group_scales = group_scales.reshape(rows, cols // 16)

    return NVFP4Quant(out, group_scales, tensor_scale)