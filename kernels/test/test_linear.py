import pytest
import torch
from quartet2.linear import Quartet_II_linear

def ref_in_fp32(H, x, y, head, w1, w2, w3):
    w1_ref = Quartet_II_linear(H, H, bias=False, device="cuda", dtype=torch.bfloat16)
    w2_ref = Quartet_II_linear(H, H, bias=False, device="cuda", dtype=torch.bfloat16)
    w3_ref = Quartet_II_linear(H, H, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        w1_ref.weight[...] = w1
        w2_ref.weight[...] = w2
        w3_ref.weight[...] = w3
    hid = w1_ref(x, disable_backward_quant=True)
    hid = torch.nn.functional.relu(hid)
    hid = w2_ref(hid, disable_backward_quant=True)
    hid = torch.nn.functional.relu(hid)
    hid = w3_ref(hid, disable_backward_quant=True)
    loss = (hid @ head - y).pow(2).mean()
    loss.backward()
    return w1_ref.weight.grad.float().detach(), w2_ref.weight.grad.float().detach(), w3_ref.weight.grad.float().detach()


def eval(res, ref):
    with torch.no_grad():
        res = res.float()
        ref = ref.float()
        quad_err = (res - ref).pow(2).mean() / ref.pow(2).mean()
        eff_bitwidth = (-torch.log2(quad_err) / 2)
        cosine = (res.flatten() @ ref.flatten()) / (ref.flatten() @ ref.flatten())
        return quad_err.item(), eff_bitwidth.item(), cosine.item()


def print_error(name, quad_err, eff_bitwidth, cosine):
    print(f"{name}: {quad_err:.4f} mrqe; {eff_bitwidth:.2f} bits; {cosine:.3f} cosine")


def evaluate_accuracy(B, T, H, steps):
    x = torch.randn((B, T, H), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((B, T, 1), device='cuda', dtype=torch.bfloat16)

    W1 = Quartet_II_linear(H, H, device='cuda', dtype=torch.bfloat16)
    W2 = Quartet_II_linear(H, H, device='cuda', dtype=torch.bfloat16)
    W3 = Quartet_II_linear(H, H, device='cuda', dtype=torch.bfloat16)

    with torch.no_grad():
        W1.weight /= (H**0.5 * W1.weight.std())
        W2.weight /= (H**0.5 * W2.weight.std())
        W3.weight /= (H**0.5 * W3.weight.std())

    head = torch.randn(H, 1, device='cuda', dtype=torch.bfloat16)


    w1_ref_grad, w2_ref_grad, w3_ref_grad = ref_in_fp32(H, x, y, head, W1.weight, W2.weight, W3.weight)

    hid = W1(x)
    hid = torch.nn.functional.relu(hid)
    hid = W2(hid)
    hid = torch.nn.functional.relu(hid)
    hid = W3(hid)
    loss = (hid @ head - y).pow(2).mean()

    W1.weight.grad = None
    W2.weight.grad = None
    W3.weight.grad = None
    R1 = torch.zeros_like(W1.weight, dtype=torch.float32)
    R2 = torch.zeros_like(W2.weight, dtype=torch.float32)
    R3 = torch.zeros_like(W3.weight, dtype=torch.float32)
    for _ in range(steps):
        loss.backward(retain_graph=True)
        R1 += W1.weight.grad.float()
        R2 += W2.weight.grad.float()
        R3 += W3.weight.grad.float()
        W1.weight.grad = None
        W2.weight.grad = None
        W3.weight.grad = None
    with torch.no_grad():
        R1 /= steps
        R2 /= steps
        R3 /= steps

        w1_err = eval(R1, w1_ref_grad)
        print_error("W1", *w1_err)

        w2_err = eval(R2, w2_ref_grad)
        print_error("W2", *w2_err)

        w3_err = eval(R3, w3_ref_grad)
        print_error("W3", *w3_err)

    return w1_err, w2_err, w3_err


torch.manual_seed(42)

@pytest.mark.parametrize(
    "steps, expected",
    [(1, (1.9, 2.5, 3.8, 0.06)),
     (4, (2.9, 3.4, 4.0, 0.02)),
     (16, (3.9, 4.5, 6.4, 0.01)),
     (64, (4.9, 5.5, 7.5, 0.01)),
     (256, (5.9, 6.5, 8.1, 0.005)),
     (1024, (6.9, 7.5, 8.9, 0.002)),
     (4096, (7.6, 8.2, 8.9, 0.001))])
def test_backward_accuracy(steps, expected):
    B = 8
    T = 16
    H = 256
    print(f"{steps=}:")
    w1e, w2e, w3e = evaluate_accuracy(B, T, H, steps)

    # bits
    assert w1e[1] >= expected[0]
    assert w2e[1] >= expected[1]
    assert w3e[1] >= expected[2]

    # cosine
    cmax = 1.0 + expected[3]
    cmin = 1.0 - expected[3]
    assert cmin < w1e[2] < cmax
    assert cmin < w2e[2] < cmax
    assert cmin < w3e[2] < cmax


def test_compile_fwd():
    W = Quartet_II_linear(128, 256, device='cuda', dtype=torch.bfloat16)
    def fwd(x): return W(x)
    torch.compile(fwd, fullgraph=True)(torch.randn(1, 128, 128, device='cuda', dtype=torch.bfloat16))


def test_result_fwd():
    W = Quartet_II_linear(128, 256, device='cuda', dtype=torch.bfloat16)
    x = torch.randn((1, 512, 128), device="cuda", dtype=torch.bfloat16)
    y = W(x)
    y_ref = x @ W.weight.T
    se, bits, cos = eval(y, y_ref)
    assert cos > 0.999
    assert bits > 2.8
