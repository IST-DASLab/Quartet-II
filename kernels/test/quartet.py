import torch

from scipy.linalg import hadamard


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5, dtype=dtype, device=device
    )


from qutlass import matmul_mxf4_bf16_tn

@torch.library.custom_op("quartet::matmul_mxf4_bf16_tn_op", mutates_args=())
def matmul_mxf4_bf16_tn_op(
    x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    return matmul_mxf4_bf16_tn(
        x.view(torch.uint8), w.view(torch.uint8), xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu), alpha
    )

@matmul_mxf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(x.shape[0], w.shape[0], dtype=torch.bfloat16)


from qutlass import fusedQuantizeMx

@torch.library.custom_op("quartet::fusedQuantizeMx_op", mutates_args=())
def fusedQuantizeMx_op(
    x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, return_mask: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if return_mask:
        return fusedQuantizeMx(x_flat, hadamard_matrix, return_mask=True)
    else:
        return fusedQuantizeMx(x_flat, hadamard_matrix, return_mask=False) + (None,)

@fusedQuantizeMx_op.register_fake
def _(x_flat, hadamard_matrix, return_mask):
    rows, cols = x_flat.shape[0], x_flat.shape[1] // 32
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4

    xh_e2m1 = torch.empty(
        x_flat.shape[0], x_flat.shape[1] // 2, dtype=torch.uint8, device=x_flat.device
    )
    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.uint8, device=x_flat.device
    )
    clip_mask = torch.empty(*x_flat.shape[:-1], x_flat.size(-1) // 8,  dtype=torch.uint8, device=x_flat.device) if return_mask else None
    return xh_e2m1, xh_e8m0, clip_mask


from qutlass import backward_t_bf16

@torch.library.custom_op("quartet::backward_t_bf16_op", mutates_args=())
def backward_t_bf16_op(
    grad_output_flat: torch.Tensor, hadamard_matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return backward_t_bf16(grad_output_flat, hadamard_matrix)

@backward_t_bf16_op.register_fake
def _(grad_output_flat, hadamard_matrix):
    xh_e2m1 = torch.empty(grad_output_flat.shape[1], grad_output_flat.shape[0] // 2,  dtype=torch.uint8, device=grad_output_flat.device)
    xh_e8m0 = torch.empty(grad_output_flat.shape[1], grad_output_flat.shape[0] // 32, dtype=torch.uint8, device=grad_output_flat.device)

    return xh_e2m1, xh_e8m0


from qutlass import backward_qt_bf16

@torch.library.custom_op("quartet::backward_qt_bf16_op", mutates_args=())
def backward_qt_bf16_op(
    x_e2m1: torch.Tensor,
    x_e8m0: torch.Tensor,
    h: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x_e2m1.dim() == 2
    return backward_qt_bf16(x_e2m1, x_e8m0, h, alpha)

@backward_qt_bf16_op.register_fake
def _(x_e2m1, x_e8m0, h, alpha):
    assert x_e2m1.dim() == 2
    xh_e2m1 = torch.empty(x_e2m1.shape[1] * 2, x_e2m1.shape[0] // 2, dtype=torch.uint8, device=h.device)
    xh_e8m0 = torch.empty(x_e8m0.shape[1] * 32, x_e8m0.shape[0] // 32, dtype=torch.uint8, device=h.device)
    return xh_e2m1, xh_e8m0


from qutlass import matmul_mxf8_bf16_tn

@torch.library.custom_op("quartet::matmul_mxf8_bf16_tn_op", mutates_args=())
def matmul_mxf8_bf16_tn_op(
    x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    return matmul_mxf8_bf16_tn(
        x, w, xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu), alpha
    )

@matmul_mxf8_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(x.shape[0], w.shape[0], dtype=torch.bfloat16)


from qutlass.utils import to_blocked

def _unpack_mask(clip_mask: torch.Tensor) -> torch.Tensor:
    clip_mask_unpacked_dq = torch.zeros(*clip_mask.shape[:-1], clip_mask.size(-1) * 8, dtype=torch.bool, device=clip_mask.device)
    for i in range(8):
        clip_mask_unpacked_dq[..., i::8] = (clip_mask >> i) & 1
    return clip_mask_unpacked_dq


FORWARD_HADAMARD_MATRIX = get_hadamard_matrix(32, dtype=torch.bfloat16, device="cuda")
BACKWARD_HADAMARD_MATRIX = get_hadamard_matrix(32, dtype=torch.bfloat16, device="cuda")


ALPHA_FWD = torch.tensor(1., device="cuda")
ALPHA_BWD = torch.tensor(1./9., device="cuda")

class QuartetGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, deterministic=True):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]

        ctx.deterministic = deterministic

        input_hf_e2m1, input_hf_e8m0, input_hf_mask = fusedQuantizeMx_op(
            input.flatten(end_dim=-2),
            FORWARD_HADAMARD_MATRIX,
            return_mask=input.requires_grad,
        )

        weight_hf_e2m1, weight_hf_e8m0, weight_hf_mask = fusedQuantizeMx_op(
            weight,
            FORWARD_HADAMARD_MATRIX,
            return_mask=input.requires_grad,
        )

        ctx.save_for_backward(input_hf_e2m1, input_hf_e8m0, input_hf_mask, weight_hf_e2m1, weight_hf_e8m0, weight_hf_mask)

        input_hf_scale_block = to_blocked(input_hf_e8m0, False)
        weight_hf_scale_block = to_blocked(weight_hf_e8m0, False)

        out = matmul_mxf4_bf16_tn_op(
            input_hf_e2m1,
            weight_hf_e2m1,
            input_hf_scale_block,
            weight_hf_scale_block,
            ALPHA_FWD,
        )
        return out.view(*input.shape[:-1], weight.size(-2))

    @staticmethod
    def backward(ctx, grad_output):
        global BACKWARD_HADAMARD_MATRIX
        input_hf_e2m1, input_hf_e8m0, input_hf_mask, weight_hf_e2m1, weight_hf_e8m0, weight_hf_mask = ctx.saved_tensors

        if not ctx.deterministic:
            BACKWARD_HADAMARD_MATRIX = BACKWARD_HADAMARD_MATRIX * (
                torch.randint(0, 2, (32,), device=BACKWARD_HADAMARD_MATRIX.device, dtype=BACKWARD_HADAMARD_MATRIX.dtype)
                * 2. - 1.
            )

        grad_output_hb_e2m1, grad_output_hb_e8m0, _ = fusedQuantizeMx_op(
            grad_output.flatten(end_dim=-2),
            BACKWARD_HADAMARD_MATRIX,
            False,
        )

        hft_weightt_hb_e2m1, hft_weightt_hb_e8m0 = backward_qt_bf16_op(weight_hf_e2m1, weight_hf_e8m0, BACKWARD_HADAMARD_MATRIX, ALPHA_FWD)
        grad_output_hb_scale_block = to_blocked(grad_output_hb_e8m0, False)
        hft_weightt_hb_scale_block = to_blocked(hft_weightt_hb_e8m0, False)
        grad_input_hf = matmul_mxf4_bf16_tn_op(
            grad_output_hb_e2m1,
            hft_weightt_hb_e2m1,
            grad_output_hb_scale_block,
            hft_weightt_hb_scale_block,
            ALPHA_BWD,
        )

        input_mask_hf = _unpack_mask(input_hf_mask)
        grad_input = (
            (grad_input_hf.view(-1, 32) * input_mask_hf.view(-1, 32).to(grad_input_hf.dtype))
            @ FORWARD_HADAMARD_MATRIX.T
        ).view(*grad_output.shape[:-1], weight_hf_e2m1.size(-1) * 2)

        grad_outputt_hb_e2m1, grad_outputt_hb_e8m0 = backward_t_bf16_op(grad_output.flatten(end_dim=-2), BACKWARD_HADAMARD_MATRIX)
        hft_inputt_hb_e2m1, hft_inputt_hb_e8m0 = backward_qt_bf16_op(input_hf_e2m1, input_hf_e8m0, BACKWARD_HADAMARD_MATRIX, ALPHA_FWD)
        grad_outputt_hb_scale_block = to_blocked(grad_outputt_hb_e8m0, False)
        hft_inputt_hb_scale_block = to_blocked(hft_inputt_hb_e8m0, False)
        grad_weight_hf = matmul_mxf4_bf16_tn_op(
            grad_outputt_hb_e2m1,
            hft_inputt_hb_e2m1,
            grad_outputt_hb_scale_block,
            hft_inputt_hb_scale_block,
            ALPHA_BWD,
        )
        
        # torch._assert(grad_weight_hf.shape == (weight_hf_e2m1.size(0), weight_hf_e2m1.size(1) * 2), f"{grad_outputt_hb_e2m1.shape=} {hft_inputt_hb_e2m1.shape=} {grad_weight_hf.shape=} {weight_hf_e2m1.shape=}")

        weight_mask_hf = _unpack_mask(weight_hf_mask)
        grad_weight = (
            (grad_weight_hf.view(-1, 32) * weight_mask_hf.view(-1, 32).to(grad_weight_hf.dtype))
            @ FORWARD_HADAMARD_MATRIX.T
        ).view(grad_output.size(-1), weight_hf_e2m1.size(-1) * 2)
        return grad_input, grad_weight, None


DUMMY_E8M0 = torch.ones(2 ** 30, dtype=torch.float8_e8m0fnu, device="cuda").view(torch.uint8)

class Fp8Gemm(torch.autograd.Function):
    @staticmethod
    def get_dummy_e8m0(x_e4m3: torch.Tensor) -> torch.Tensor:
        x_e8m0 = DUMMY_E8M0[:x_e4m3.numel() // 32].view(*x_e4m3.shape[:-1], x_e4m3.size(-1) // 32)
        return x_e8m0

    @staticmethod
    def mm_fp8(a_e4m3: torch.Tensor, b_e4m3: torch.Tensor) -> torch.Tensor:
        c_bf16 = matmul_mxf8_bf16_tn_op(a_e4m3, b_e4m3, Fp8Gemm.get_dummy_e8m0(a_e4m3), Fp8Gemm.get_dummy_e8m0(b_e4m3), ALPHA_FWD)
        return c_bf16

    @staticmethod
    def forward(ctx, input, weight):
        input_e4m3 = input.flatten(end_dim=-2).to(dtype=torch.float8_e4m3fn)
        weight_e4m3 = weight.to(dtype=torch.float8_e4m3fn)
        ctx.save_for_backward(input_e4m3, weight_e4m3)

        return Fp8Gemm.mm_fp8(
            input_e4m3,
            weight_e4m3,
        ).view(*input.shape[:-1], weight.size(-2))

    @staticmethod
    def backward(ctx, grad_output):
        input_e4m3, weight_e4m3 = ctx.saved_tensors

        grad_output_e4m3 = grad_output.flatten(end_dim=-2).to(dtype=torch.float8_e4m3fn)
        
        grad_input = Fp8Gemm.mm_fp8(
            grad_output_e4m3,
            weight_e4m3.T.contiguous(),
        ).view(*grad_output.shape[:-1], weight_e4m3.size(-1))

        grad_outputt_e4m3 = grad_output.flatten(end_dim=-2).to(dtype=torch.float8_e4m3fn)
        grad_weight = Fp8Gemm.mm_fp8(
            grad_outputt_e4m3.T.contiguous(),
            input_e4m3.T.contiguous(),
        ).view(grad_output.size(-1), weight_e4m3.size(-1))

        return grad_input, grad_weight
