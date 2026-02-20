import sys
import torch
import nvtx
from flashinfer import autotune
from quartet2.linear import Quartet_II_linear

try:
    from quartet import QuartetGemm, Fp8Gemm
    _HAS_QUARTET_1 = True
except Exception:
    _HAS_QUARTET_1 = False

# ensure we're not hitting compile limits
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


def bench_shape(
        in_dim, out_dim, batch_size, seq_len,
        weight_dtype=torch.bfloat16, act_dtype=torch.bfloat16, device='cuda',
        warmup=10, rep=100,
):
    import triton

    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=act_dtype)
    amax = x.abs().max().float()

    linear = Quartet_II_linear(in_dim, out_dim, four_over_six=True, device=device, dtype=weight_dtype)
    with torch.no_grad():
        linear.abs_max = linear.weight.abs().max().float()

    # autotune
    with autotune(True):
        linear(x, input_abs_max=amax)

    # Forward
    torch.set_grad_enabled(False)

    ms = triton.testing.do_bench(
        lambda: linear(x, input_abs_max=amax), warmup=warmup, rep=rep,
    )
    forward_time = ms

    # Forward+Backward
    grad = torch.randn_like(linear(x))
    torch.set_grad_enabled(True)

    def forward_backward(x, grad):
        output = linear(x, input_abs_max=amax)
        output.backward(grad)

    x.requires_grad_(True)
    ms = triton.testing.do_bench(
        lambda: forward_backward(x, grad), warmup=warmup, rep=rep,
    )
    total_time = ms

    return {
        "forward_ms": forward_time,
        "total_ms": total_time,
    }

def bench_shape_bf16(
        in_dim, out_dim, batch_size, seq_len,
        weight_dtype=torch.bfloat16, act_dtype=torch.bfloat16, device='cuda',
        warmup=10, rep=100,
):
    import triton

    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=act_dtype)

    linear = torch.nn.Linear(in_dim, out_dim, device=device, dtype=weight_dtype)
    # Forward
    torch.set_grad_enabled(False)

    ms = triton.testing.do_bench(
        lambda: linear(x), warmup=warmup, rep=rep,
    )
    forward_time = ms

    # Forward+Backward
    grad = torch.randn_like(linear(x))
    torch.set_grad_enabled(True)

    x.requires_grad_(True)
    def forward_backward(x, grad):
        output = linear(x)
        output.backward(grad)

    ms = triton.testing.do_bench(
        lambda: forward_backward(x, grad), warmup=warmup, rep=rep,
    )
    total_time = ms

    return {
        "forward_ms": forward_time,
        "total_ms": total_time,
    }


def bench_shape_fpquant(
        in_dim, out_dim, batch_size, seq_len,
        weight_dtype=torch.bfloat16, act_dtype=torch.bfloat16, device='cuda',
        warmup=10, rep=100,
):
    import triton

    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=act_dtype, requires_grad=True)
    w = torch.randn(out_dim, in_dim, device=device, dtype=weight_dtype, requires_grad=True)

    linear = lambda x: QuartetGemm.apply(x, w)

    # Forward
    torch.set_grad_enabled(False)

    ms = triton.testing.do_bench(
        lambda: linear(x), warmup=warmup, rep=rep,
    )
    forward_time = ms

    # Forward+Backward
    grad = torch.randn_like(linear(x))
    torch.set_grad_enabled(True)

    def forward_backward(x, grad):
        output = linear(x)
        output.backward(grad)

    x.requires_grad_(True)
    ms = triton.testing.do_bench(
        lambda: forward_backward(x, grad), warmup=warmup, rep=rep,
    )
    total_time = ms

    return {
        "forward_ms": forward_time,
        "total_ms": total_time,
    }


def bench_shape_fp8(
        in_dim, out_dim, batch_size, seq_len,
        weight_dtype=torch.bfloat16, act_dtype=torch.bfloat16, device='cuda',
        warmup=10, rep=100,
):
    import triton
    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=act_dtype, requires_grad=True)
    w = torch.randn(out_dim, in_dim, device=device, dtype=weight_dtype, requires_grad=True)

    linear = lambda x: Fp8Gemm.apply(x, w)

    # Forward
    torch.set_grad_enabled(False)

    ms = triton.testing.do_bench(
        lambda: linear(x), warmup=warmup, rep=rep,
    )
    forward_time = ms

    # Forward+Backward
    grad = torch.randn_like(linear(x))
    torch.set_grad_enabled(True)

    def forward_backward(x, grad):
        output = linear(x)
        output.backward(grad)

    x.requires_grad_(True)
    ms = triton.testing.do_bench(
        lambda: forward_backward(x, grad), warmup=warmup, rep=rep,
    )
    total_time = ms

    return {
        "forward_ms": forward_time,
        "total_ms": total_time,
    }


def bench_shape_pure_matmul(
        in_dim, out_dim, batch_size, seq_len,
        weight_dtype=torch.bfloat16, act_dtype=torch.bfloat16, device='cuda',
        warmup=10, rep=100,
):
    import triton
    from quartet2.quant import quant_fp4, rht128_quant_eden, rht128_requant, NVFP4QuantMode
    from quartet2.linear import _fp4_mm

    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=act_dtype)
    linear = Quartet_II_linear(in_dim, out_dim, four_over_six=True, device=device, dtype=weight_dtype)
    weight = linear.weight
    input_amax = x.abs().max().float()
    weight_amax = linear.weight.abs().max().float()

    flat_input = x.reshape(-1, x.shape[-1])
    input_fp4 = quant_fp4(flat_input, amax=input_amax, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX )
    weight_fp4 = quant_fp4(weight, amax=weight_amax, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX )

    # Forward
    torch.set_grad_enabled(False)

    ms = triton.testing.do_bench(
        lambda:  _fp4_mm(
            input_fp4.fp4, weight_fp4.fp4,
            input_fp4.micro_scales, weight_fp4.micro_scales,
            alpha=input_fp4.tensor_scale * weight_fp4.tensor_scale), warmup=warmup, rep=rep,
    )
    forward_time = ms

    # Forward+Backward
    grad = torch.randn_like(linear(x))
    torch.set_grad_enabled(True)

    def forward_backward(x, grad):
        # quick and dirty proxy: three forwards ~ fwd+bwd
        _fp4_mm(
            input_fp4.fp4, weight_fp4.fp4,
            input_fp4.micro_scales, weight_fp4.micro_scales,
            alpha=input_fp4.tensor_scale * weight_fp4.tensor_scale)
        _fp4_mm(
            input_fp4.fp4, weight_fp4.fp4,
            input_fp4.micro_scales, weight_fp4.micro_scales,
            alpha=input_fp4.tensor_scale * weight_fp4.tensor_scale)
        _fp4_mm(
            input_fp4.fp4, weight_fp4.fp4,
            input_fp4.micro_scales, weight_fp4.micro_scales,
            alpha=input_fp4.tensor_scale * weight_fp4.tensor_scale)

    x.requires_grad_(True)
    ms = triton.testing.do_bench(
        lambda: forward_backward(x, grad), warmup=warmup, rep=rep,
    )
    total_time = ms

    return {
        "forward_ms": forward_time,
        "total_ms": total_time,
    }


from tqdm.auto import tqdm, trange

BATCH_SIZE = 16
SEQ_LEN = 2048

shapes = {
    # Q K V Down Up Gate Down
    #"100M": [(1024 * 3, 1024), (1024, 1024), (2816 * 2, 1024), (1024, 2816)],
    #"800M": [(2048 * 3, 2048), (2048, 2048), (5632 * 2, 2048), (2048, 5632)],
    "3B": [(3072 * 3, 3072), (3072, 3072), (8192 * 2, 3072), (3072, 8192)],
    "7B": [(4096 * 3, 4096), (4096, 4096), (11008 * 2, 4096), (4096, 11008)],
    "22B": [(6144 * 3, 6144), (6144, 6144), (16384 * 2, 6144), (6144, 16384)],
    "52B": [(8192 * 3, 8192), (8192, 8192), (22016 * 2, 8192), (8192, 22016)],
}

t = torch.cuda.get_device_properties(0).total_memory
if t < 12e9:
    del shapes["7B"]
if t < 24e9:
    del shapes["22B"]
if t < 32e9:
    del shapes["52B"]

print(f"Total memory: {t/1e9:.2f} GB -> using shapes: {', '.join(shapes.keys())}")

layers = {}
inputs = {}
input_amax = {}
outputs = {}
grads = {}
print(f"Creating layers")
for size, model_shapes in shapes.items():
    for shape in model_shapes:
        if shape not in layers:
            layers[shape] = Quartet_II_linear(*shape, four_over_six=True, device='cuda', dtype=torch.bfloat16)
            inputs[shape] = torch.randn(BATCH_SIZE, SEQ_LEN, shape[0], device='cuda', dtype=torch.bfloat16)
            with torch.no_grad():
                layers[shape].weight_abs_max = layers[shape].weight.abs().max().float()
                input_amax[shape] = inputs[shape].abs().max().float()
            torch.set_grad_enabled(False)
            layers[shape](inputs[shape])

torch.cuda.synchronize()

# profiling: use this when running with nsys
profiling = sys.argv[1] == "--profile" if len(sys.argv) > 1 else False
if profiling:
    with nvtx.annotate("ProfilingForward", color="green"):
        for size, model_shapes in shapes.items():
            with nvtx.annotate(f"{size}"):
                for shape in model_shapes:
                    with nvtx.annotate(f"Shape={shape}"):
                        layers[shape](inputs[shape], input_abs_max=input_amax[shape])
    torch.cuda.synchronize()
    print("Profiling done")

    outputs = {}
    grads = {}
    for size, model_shapes in shapes.items():
        for shape in model_shapes:
            if shape not in outputs:
                torch.set_grad_enabled(True)
                out = layers[shape](inputs[shape], input_abs_max=input_amax[shape])
                grad = torch.randn_like(out)
                grads[shape] = grad
                out.backward(grad)

                out = layers[shape](inputs[shape])
                outputs[shape] = out

    torch.cuda.synchronize()

    with nvtx.annotate("ProfilingBackward", color="green"):
        for size, model_shapes in shapes.items():
            with nvtx.annotate(f"{size}"):
                for shape in model_shapes:
                    torch.cuda.synchronize()
                    with nvtx.annotate(f"Shape={shape}"):
                        outputs[shape].backward(grads[shape])
                        torch.cuda.synchronize()
    print("Profiling done")
    exit(0)

shape_to_result_fp4 = {}
shape_to_result_bf16 = {}
shape_to_result_fpquant = {}
shape_to_result_fp8 = {}
shape_to_result_mm = {}
shape_to_flops = {}
for size, model_shapes in tqdm(shapes.items(), desc="Iterating model sizes"):
    for shape in tqdm(model_shapes, desc="Iterating shapes", leave=False):
        if shape not in shape_to_result_fp4:
            result = bench_shape(
                shape[1], shape[0], BATCH_SIZE, SEQ_LEN,
            )
            shape_to_result_fp4[shape] = result
            result = bench_shape_bf16(
                shape[1], shape[0], BATCH_SIZE, SEQ_LEN,
            )
            shape_to_result_bf16[shape] = result
            if _HAS_QUARTET_1:
                result = bench_shape_fpquant(
                    shape[1], shape[0], BATCH_SIZE, SEQ_LEN,
                )
                shape_to_result_fpquant[shape] = result
                result = bench_shape_fp8(
                    shape[1], shape[0], BATCH_SIZE, SEQ_LEN,
                )
                shape_to_result_fp8[shape] = result
            result = bench_shape_pure_matmul(
                shape[1], shape[0], BATCH_SIZE, SEQ_LEN,
            )
            shape_to_result_mm[shape] = result
            shape_to_flops[shape] = 6 * shape[1] * shape[0] * BATCH_SIZE * SEQ_LEN



print("Baseline:")
for size, model_shapes in shapes.items():
    forward_latency = sum(shape_to_result_bf16[shape]['forward_ms'] for shape in model_shapes)
    total_latency = sum(shape_to_result_bf16[shape]['total_ms'] for shape in model_shapes)
    total_flops = sum(shape_to_flops[shape] for shape in model_shapes)
    print(f"\t{size:5}:  {forward_latency:6.2f} ms forward, {total_latency:6.2f} ms forward+backward, {total_flops/1e12:5.2f} TFLOP | {total_flops / (total_latency/1000)/1e12:6.1f} TFLOp/s")

print()

for size, model_shapes in shapes.items():
    forward_latency = sum(shape_to_result_fp4[shape]['forward_ms'] for shape in model_shapes)
    total_latency = sum(shape_to_result_fp4[shape]['total_ms'] for shape in model_shapes)
    forward_latency_bf16 = sum(shape_to_result_bf16[shape]['forward_ms'] for shape in model_shapes)
    total_latency_bf16 = sum(shape_to_result_bf16[shape]['total_ms'] for shape in model_shapes)
    if _HAS_QUARTET_1:
        forward_latency_fpquant = sum(shape_to_result_fpquant[shape]['forward_ms'] for shape in model_shapes)
        total_latency_fpquant = sum(shape_to_result_fpquant[shape]['total_ms'] for shape in model_shapes)
        forward_latency_fp8 = sum(shape_to_result_fp8[shape]['forward_ms'] for shape in model_shapes)
        total_latency_fp8 = sum(shape_to_result_fp8[shape]['total_ms'] for shape in model_shapes)
    forward_latency_mm = sum(shape_to_result_mm[shape]['forward_ms'] for shape in model_shapes)
    total_latency_mm = sum(shape_to_result_mm[shape]['total_ms'] for shape in model_shapes)
    total_flops = sum(shape_to_flops[shape] for shape in model_shapes)


    print(f"{size}:")
    print(f"\tnvfp4: {forward_latency:6.2f} ms forward, {total_latency:6.2f} ms forward+backward <=> {total_flops / (total_latency/1000)/1e12:6.1f} TFLOp/s")
    print(f"\t gemm: {forward_latency_mm:6.2f} ms forward, {total_latency_mm:6.2f} ms forward+backward <=> {total_flops / (total_latency_mm/1000)/1e12:6.1f} TFLOp/s")
    if _HAS_QUARTET_1:
        print(f"\tmxfp4: {forward_latency_fpquant:6.2f} ms forward, {total_latency_fpquant:6.2f} ms forward+backward")
        print(f"\tmxfp8: {forward_latency_fp8:6.2f} ms forward, {total_latency_fp8:6.2f} ms forward+backward")
    print(f"\tnvfp4: {100*forward_latency_bf16/forward_latency:6.2f}%   forward, {100*total_latency_bf16/total_latency:6.2f}%   forward+backward")
    if _HAS_QUARTET_1:
        print(f"\tmxfp4: {100*forward_latency_bf16/forward_latency_fpquant:6.2f}%   forward, {100*total_latency_bf16/total_latency_fpquant:6.2f}%   forward+backward")
        print(f"\tmxfp8: {100*forward_latency_bf16/forward_latency_fp8:6.2f}%   forward, {100*total_latency_bf16/total_latency_fp8:6.2f}%   forward+backward")
