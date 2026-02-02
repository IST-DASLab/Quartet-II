import torch
import qutlass

from scipy.linalg import hadamard
from .quant import quant_fp4, quant_had_eden, dequant_tp_had_eden
import nvtx
import contextlib

def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
        )


def nvtx_annotate(name: str, color: str = "green"):
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    else:
        return nvtx.annotate(name, color=color)


def rerotate_hadamard(hadamard_matrix):
    signs = torch.randint(
            0, 2, (hadamard_matrix.size(0),),
            device=hadamard_matrix.device,
            dtype=hadamard_matrix.dtype
        ) * 2 - 1
    return hadamard_matrix * signs[None, :] # NOTE: rerotate along last dim, inner dim for TN GEMM


@torch.library.custom_op("clover::fp4_mm", mutates_args=())
def _fp4_mm(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return qutlass.matmul_nvf4_bf16_tn(
        x_fp4, w_fp4,
        x_mx, w_mx,
        alpha=alpha)


@_fp4_mm.register_fake
def _fp4_mm_fake(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.empty((x_fp4.shape[0], w_fp4.shape[0]), device=x_fp4.device, dtype=torch.bfloat16)


def to_blocked(input_matrix) -> torch.Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    This function is copied from qutlass, but compatible with torch.compile.

    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """
    rows, cols = input_matrix.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # Note: No second argument to assert, that broke torch.compile
    assert (rows, cols) == (padded_rows, padded_cols)

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


@torch.compile
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
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq[..., None]).flatten(
        start_dim=-2
    ) * alpha
    return x_dq.to(torch.bfloat16)

@torch.compile(dynamic=False)
def abs_max(x):
    return x.abs().max().to(torch.float32)

class Quartet_II_fn(torch.autograd.Function):
    group_size = 16

    #@torch.compile(dynamic=False)
    @staticmethod
    def forward(ctx, input, weight, had, four_over_six: bool, disable_backward_quant: bool = False, weight_amax: torch.Tensor = None, input_amax: torch.Tensor = None, scratch_amax: torch.Tensor = None):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.disable_backward_quant = disable_backward_quant
        ctx.scratch_amax = scratch_amax
        assert input.dtype == torch.bfloat16

        forward_scale_override = 1.0

        flat_input = input.reshape(-1, input.shape[-1])

        with nvtx_annotate("Abs-max", color="red"):
            if input_amax is None:
                input_amax = abs_max(flat_input)
            if weight_amax is None:
                weight_amax = abs_max(weight)

        with nvtx_annotate("Quant", color="yellow"):
            input_fp4 = quant_fp4(flat_input, amax=input_amax, scale_override=forward_scale_override, four_over_six=four_over_six)
            weight_fp4 = quant_fp4(weight, amax=weight_amax, scale_override=forward_scale_override, four_over_six=four_over_six)
        # TODO save quantized for requant kernels
        ctx.save_for_backward(input_fp4.fp4, input_fp4.micro_scales, input_fp4.tensor_scale,
                              weight_fp4.fp4, weight_fp4.micro_scales, weight_fp4.tensor_scale, had)
        with nvtx_annotate("Matmul", color="blue"):
            res = _fp4_mm(
                input_fp4.fp4, weight_fp4.fp4,
                to_blocked(input_fp4.micro_scales), to_blocked(weight_fp4.micro_scales),
                alpha=input_fp4.tensor_scale * weight_fp4.tensor_scale)

        return res.reshape(ctx.batch, ctx.seq, ctx.out_dim)

    #@torch.compile(dynamic=False)
    @staticmethod
    def backward(ctx, grad_output):
        # Load ctx and reshape
        xfp4, xs, xm, wfp4, ws, wm, had = ctx.saved_tensors
        backward_scale_override = (17 / 16) * 0.93

        # Re-randomize the rotation
        had = rerotate_hadamard(had)
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.disable_backward_quant:
            xr = _dq_fp4(xfp4, xs, xm)
            wr = _dq_fp4(wfp4, ws, wm)
            grad_input = flat_grad_output @ wr
            grad_weight = flat_grad_output.T @ xr
            return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None

        # EW
        with nvtx_annotate("Quant", color="yellow"):
            e_ht_fp4, e_ht_ms, e_ht_ts = quant_had_eden(x=flat_grad_output, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
            wt_ht_fp4, wt_ht_ms, wt_ht_ts = dequant_tp_had_eden(x=wfp4, x_group_scales=ws, x_tensor_scale=wm, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
        with nvtx_annotate("Matmul", color="blue"):
            grad_input = _fp4_mm(e_ht_fp4, wt_ht_fp4, to_blocked(e_ht_ms), to_blocked(wt_ht_ms), alpha=e_ht_ts*wt_ht_ts)

        # EtX
        with nvtx_annotate("Quant", color="yellow"):
            et_ht_fp4, et_ht_ms, et_ht_ts = quant_had_eden(x=flat_grad_output, h=had, scale_override=backward_scale_override, transpose=True, scratch_amax=ctx.scratch_amax)
            xt_ht_fp4, xt_ht_ms, xt_ht_ts = dequant_tp_had_eden(x=xfp4, x_group_scales=xs, x_tensor_scale=xm, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
        with nvtx_annotate("Matmul", color="blue"):
            grad_weight = _fp4_mm(et_ht_fp4, xt_ht_fp4, to_blocked(et_ht_ms), to_blocked(xt_ht_ms), alpha=et_ht_ts*xt_ht_ts)
        return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None


class Quartet_II_linear(torch.nn.Linear):
    def __init__(self, *args, four_over_six=False, dtype=torch.bfloat16, **kwargs):
        super().__init__(*args, dtype=dtype, **kwargs)
        assert dtype == torch.bfloat16
        self.four_over_six = four_over_six
        self.weight_abs_max = None
        self.register_buffer("had", get_hadamard_matrix(128, self.weight.dtype, self.weight.device))
        self.register_buffer("scratch_amax", torch.empty((), dtype=torch.int32, device=self.weight.device))

    def forward(self, x, disable_backward_quant=False, input_abs_max=None):
        return Quartet_II_fn.apply(x, self.weight[...], self.had, self.four_over_six, disable_backward_quant, self.weight_abs_max, input_abs_max, self.scratch_amax)
