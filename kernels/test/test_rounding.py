import pytest
import quartet2.quant
import torch

torch.random.manual_seed(42)


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
    scales_dq = scales_dq.unflatten(dim=-1, sizes=(-1, 1))
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq).flatten(
        start_dim=-2
    ) * alpha
    return x_dq


@torch.compile(fullgraph=True)
def quant_fp4(x, four_over_six=False):
    amax = torch.max(torch.abs(x)).to(torch.float32)
    q, s, t = quartet2.quant.quant_fp4(x, 1.0, amax, four_over_six=four_over_six)
    return _dq_fp4(q, s, t)


@torch.compile(fullgraph=True)
def quant_with_transform(x, transpose=False):
    e = torch.eye(128, dtype=torch.bfloat16, device=x.device)
    h = torch.empty((128, 128), dtype=torch.bfloat16, device=x.device)
    #  output is swizzled, so we need to prepare h accordingly
    for j in range(0, 128, 16):
        order = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
        for i in range(16):
            h[:, j + i] = e[:, j+order[i]]
    q, s, g = quartet2.quant.quant_had_eden(x=x, h=h, transpose=transpose)
    return _dq_fp4(q, s, g)


@pytest.mark.parametrize("four_over_six,expected", [(False, 3.38), (True, 3.52)])
def test_cvt_nvfp4(four_over_six, expected):
    x = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
    dq = quant_fp4(x.to(torch.bfloat16), four_over_six)
    quad_err = ((x - dq).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1)).mean()
    eff_bitwidth = (-torch.log2(quad_err) / 2).item()

    assert eff_bitwidth > expected


@pytest.mark.parametrize("transpose,expected", [(False, 3.37), (True, 3.37)])
def test_eden_identity(transpose, expected):
    x = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
    dq = quant_with_transform(x.to(torch.bfloat16), transpose=transpose)
    if transpose:
        x = x.transpose(-1, -2)
    quad_err = ((x - dq).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1)).mean()
    eff_bitwidth = (-torch.log2(quad_err) / 2).item()

    assert eff_bitwidth > expected

