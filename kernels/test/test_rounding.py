import pytest
import quartet2.quant
import torch
from quartet2.linear import unblock, get_hadamard_matrix
from quartet2.quant import NVFP4QuantMode
from quartet2.rht import swizzle_hadamard

torch.random.manual_seed(42)

def reference_transform(x, h, transpose=False):
    if transpose:
        x = x.T
    r = x.reshape(-1, 128) @ h.T
    return r.reshape(x.shape)


def _dq_fp4(x_e2m1: torch.Tensor, x_e4m3: torch.Tensor, alpha: float):
    device = x_e2m1.device

    x_e2m1_i32 = x_e2m1.view(dtype=torch.uint8).to(dtype=torch.int32)
    x_e2m1_unpacked = torch.stack(
        [x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1
    ).flatten(start_dim=-2)

    grid_dq = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    x_fp4_dq = grid_dq[x_e2m1_unpacked]
    scales_dq = x_e4m3.to(torch.float32)
    scales_dq = unblock(scales_dq, x_e2m1.shape[0], x_e2m1.shape[1] * 2)
    scales_dq = scales_dq.unflatten(dim=-1, sizes=(-1, 1))
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq).flatten(
        start_dim=-2
    ) * alpha
    return x_dq


@torch.compile(fullgraph=True)
def quant_fp4(x, mode):
    amax = torch.max(torch.abs(x)).to(torch.float32)
    seed = torch.tensor(42, dtype=torch.int64)
    q, s, t = quartet2.quant.quant_fp4(x, scale_override=1.0, amax=amax, mode=mode, seed=seed)
    return _dq_fp4(q, s, t)


@torch.compile(fullgraph=True)
def quant_with_transform(x: torch.Tensor, transpose=False):
    e = torch.eye(128, dtype=torch.bfloat16, device=x.device)
    q, s, g = quartet2.quant.quant_had_eden(x=x, h=e, transpose=transpose)
    return _dq_fp4(q, s, g)


@torch.compile(fullgraph=True)
def quant_with_hadamard(x: torch.Tensor, h: torch.Tensor, transpose=False):
    q, s, g = quartet2.quant.rht128_quant_eden(x=x, h=h[:16, :], transpose=transpose)
    return _dq_fp4(q, s, g)


@torch.compile(fullgraph=True)
def requant_with_transform(x: torch.Tensor):
    # original transform
    e = torch.eye(128, dtype=torch.bfloat16, device=x.device)
    q, s, g = quartet2.quant.quant_had_eden(x=x, h=e, transpose=False)
    dq = _dq_fp4(q, s, g).to(torch.bfloat16)

    # option 1: transpose and quantize again
    ref = quartet2.quant.quant_had_eden(x=dq, h=e, transpose=True)
    ref = _dq_fp4(ref[0], ref[1], ref[2])

    # option 2: directly requantize
    out = quartet2.quant.dequant_tp_had_eden(x=q, h=e, x_group_scales=s, x_tensor_scale=g)
    out = _dq_fp4(out[0], out[1], out[2])
    return out, ref


@pytest.mark.parametrize("mode,expected", [(NVFP4QuantMode.RNE, 3.38), (NVFP4QuantMode.FOUR_SIX, 3.52), (NVFP4QuantMode.EDEN, 3.32)])
def test_cvt_nvfp4(mode, expected):
    x = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
    dq = quant_fp4(x.to(torch.bfloat16), mode)
    quad_err = ((x - dq).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1)).mean()
    eff_bitwidth = (-torch.log2(quad_err) / 2).item()

    assert eff_bitwidth > expected


@pytest.mark.parametrize("transpose,expected", [(False, 3.37), (True, 3.37)])
def test_eden_identity(transpose, expected):
    x = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
    dq = quant_with_transform(x, transpose=transpose)
    if transpose:
        x = x.transpose(-1, -2)
    quad_err = ((x - dq).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1)).mean()
    eff_bitwidth = (-torch.log2(quad_err) / 2).item()

    assert eff_bitwidth > expected


def test_eden_requantize():
    x = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
    out, ref = requant_with_transform(x)

    quad_err = ((out - ref).pow(2).sum(dim=-1) / ref.pow(2).sum(dim=-1)).mean()
    eff_bitwidth = (-torch.log2(quad_err) / 2).item()
    assert eff_bitwidth > 4.2


@pytest.mark.parametrize("transpose", [False, True])
def test_quant_with_hadamard(transpose):
    x = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
    h = get_hadamard_matrix(128, dtype=torch.bfloat16, device=x.device)
    xh = reference_transform(x, swizzle_hadamard(h), transpose=transpose)
    dq = quant_with_hadamard(x, h, transpose=transpose)
    quad_err = ((xh - dq).pow(2).sum(dim=-1) / xh.pow(2).sum(dim=-1)).mean()
    eff_bitwidth = (-torch.log2(quad_err) / 2).item()
    assert eff_bitwidth > 3.2
