from collections import namedtuple
import torch
from . import _quartet2


# torch-compile compatible wrapping
@torch.library.custom_op("quartet2::transform_128", mutates_args=("out",))
def _group_transform_128(out: torch.Tensor, x: torch.Tensor, h: torch.Tensor, transpose: bool) -> None:
    _quartet2.group_transform_128(out, h, x, transpose)


@torch.library.custom_op("quartet2::transform_rht128", mutates_args=("out",))
def _transform_rht128(out: torch.Tensor, x: torch.Tensor, h: torch.Tensor, transpose: bool) -> None:
    _quartet2.transform_rht128(out, h, x, transpose)



def transform_128(*, out: torch.Tensor = None, h: torch.Tensor, x: torch.Tensor, transpose: bool = False):
    assert h.dim() == 2 and h.shape == (128, 128)
    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert h.is_cuda
    assert h.is_contiguous()

    original_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    rows, cols = x.shape
    if transpose:
        # TODO transpose with more dims doesn't pass tests
        assert len(original_shape) == 2
        cols, rows = rows, cols
    if out is None:
        out = torch.empty((rows, cols), dtype=torch.bfloat16, device=x.device)
    _group_transform_128(out, x, h, transpose)
    out = out.reshape(original_shape)
    if transpose:
        out = out.reshape(out.T.shape)
    return out

def transform_rht128(*, out: torch.Tensor = None, h: torch.Tensor, x: torch.Tensor, transpose: bool = False):
    assert h.dim() == 2 and h.shape == (16, 128)
    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert h.is_cuda
    assert h.is_contiguous()

    original_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    rows, cols = x.shape
    if transpose:
        # TODO transpose with more dims doesn't pass tests
        assert len(original_shape) == 2
        cols, rows = rows, cols
    if out is None:
        out = torch.empty((rows, cols), dtype=torch.bfloat16, device=x.device)
    _transform_rht128(out, x, h, transpose)
    out = out.reshape(original_shape)
    if transpose:
        out = out.reshape(out.T.shape)
    return out


def swizzle_hadamard(hadamard_matrix):
    """
    This function implements the swizzle pattern used for rht128.
    Given a "standard" hadamard matrix generated using Sylvester's construction,
    with potentially random sign-flips, the following identity holds:
    ```
    transform_128(x, swizzle_hadamard(h)) == transform_rht128(x, h[:16, :])
    ```

    Note: This function exists purely for debugging and testing, it is *not* efficient.
    """
    r = torch.empty_like(hadamard_matrix)
    reorder = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
    for k in range(0, 128, 64):
        for l in range(4):
            for j in range(16):
                r[k + 16*l + j, :] = hadamard_matrix[k + 4*l + reorder[j],  :]
    return r

