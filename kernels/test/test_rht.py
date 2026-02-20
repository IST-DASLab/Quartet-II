import pytest
import torch
from quartet2.rht import transform_128, transform_rht128, swizzle_hadamard
from quartet2.quant import quant_had_eden, rht128_quant_eden
from quartet2.linear import get_hadamard_matrix

torch.random.manual_seed(42)


def reference_transform(x, h, transpose=False):
    if transpose:
        x = x.T
    r = x.reshape(-1, 128) @ h.T
    return r.reshape(x.shape)


@pytest.mark.parametrize("shape", [(128, 128), (256, 128), (128, 1024), (1024, 640)])
@pytest.mark.parametrize("transpose", [False, True])
def test_transform(shape, transpose):
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    h = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    r = reference_transform(x, h, transpose=transpose)
    r2 = transform_128(h=h, x=x, transpose=transpose)
    assert torch.allclose(r, r2)


def rerotate_hadamard(hadamard_matrix):
    signs = torch.randint(
        0, 2, (hadamard_matrix.size(0),),
        device=hadamard_matrix.device,
        dtype=hadamard_matrix.dtype
    ) * 2 - 1
    return hadamard_matrix * signs[None, :] # NOTE: rerotate along last dim, inner dim for TN GEMM

# actual hadamard
@pytest.mark.parametrize("shape", [(128, 128), (256, 128), (128, 1024), (1024, 640)])
@pytest.mark.parametrize("transpose", [False, True])
def test_transform_hadamard(shape, transpose):
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    h = rerotate_hadamard(get_hadamard_matrix(128, torch.bfloat16, 'cuda') * 128**0.5)
    r = reference_transform(x, swizzle_hadamard(h), transpose=transpose)
    h_slice = h[:16, :]
    r2 = transform_rht128(h=h_slice, x=x, transpose=transpose)

    assert torch.allclose(r, r2, rtol=1e-3, atol=5e-5)


# check equivalence of implementations
@pytest.mark.parametrize("shape", [(128, 128), (256, 128), (128, 1024), (1024, 640)])
@pytest.mark.parametrize("transpose", [False, True])
def test_transform_hadamard_eden(shape, transpose):
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    h = rerotate_hadamard(get_hadamard_matrix(128, torch.bfloat16, 'cuda') * 128**0.5)
    rq, rs, rg = quant_had_eden(x=x, h=swizzle_hadamard(h), transpose=transpose, seed=torch.tensor(42, dtype=torch.int64))
    h_slice = h[:16, :]
    tq, ts, tg = rht128_quant_eden(h=h_slice, x=x, transpose=transpose, seed=torch.tensor(42, dtype=torch.int64))
    assert rg.item() == tg.item()
    assert torch.allclose(rs.float(), ts.float(), atol=0, rtol=0)
    assert torch.allclose(rq, tq, atol=0, rtol=0)
