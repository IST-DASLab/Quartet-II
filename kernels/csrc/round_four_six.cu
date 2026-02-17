// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>

#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;
using fp32x8 = GenericVector<float, 8>;
using fp4x8 = GenericVector<unsigned char, 4>;

struct QuantResult {
    fp4x8 bits;
    float scale;
    __nv_fp8_e4m3 fp8s;
};

__device__ __forceinline__ QuantResult quantize(float abs_max, float inv_val_max, float scale, bf16x8& x) {
    float s_group = abs_max * inv_val_max;
    float inv_scale = reciprocal_approximate_ftz(scale);
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group * inv_scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0) s_round_fp8 = 1.f;

    float factor = reciprocal_approximate_ftz(s_round_fp8 * scale);
    float2 factor2 = {factor, factor};
    fp4x8 result;
    for (int k = 0; k < bf16x8::size; k += 2) {
        float2 src = make_float2(static_cast<float>(x[k+0]), static_cast<float>(x[k+1]));
        float2 scaled = __fmul2_rn(src, factor2);
        unsigned char bits = __nv_cvt_float2_to_fp4x2(scaled, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
        result[k/2] = bits;
    }

    return QuantResult{result, s_round_fp8, s_as_fp8};
}

__forceinline__ __device__ float quant_error(bf16x8 x, const QuantResult& q, float scale) {
    const float descale = static_cast<float>(q.fp8s) * scale;
    float2 sum = {0.f, 0.f};
    const float2 dsv = {-descale, -descale};
    for (int i = 0; i < 4; ++i) {
        float2 dq = __nv_cvt_fp4x2_to_float2(q.bits[i]);
        float2 xv = {static_cast<float>(x[2*i+0]), static_cast<float>(x[2*i+1])};
        float2 d = __ffma2_rn(dq, dsv, xv);
        sum = __ffma2_rn(d, d, sum);
    }
    float local_error = sum.x + sum.y;
    local_error += __shfl_xor_sync(0xffffffff, local_error, 1);
    return local_error;
}


template<float... Others>
struct get_candidate_helper;

template<float Value, float... Others>
struct get_candidate_helper<Value, Others...> {
    static constexpr __forceinline__ __device__ float get_inv(int i) {
        constexpr float inv_val = 1.f / Value;
        if (i == 0) return inv_val;
        return get_candidate_helper<Others...>::get_inv(i - 1);
    }
};

template<>
struct get_candidate_helper<> {
    static constexpr __forceinline__ __device__ float get_inv(int i) {
        __builtin_unreachable();
    }
};

template<float... Candidates>
__global__ void four_six_fp4_kernel(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, int nvecs, int cols) {
    constexpr int NumCandidates = sizeof...(Candidates);
    float global_abs_max = *amax_ptr;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= nvecs) return;

    bf16x8 x = bf16x8::load(x_ptr + 8 * idx);

    constexpr float inv_scales_max = NumCandidates > 1 ? 1.f / 256.f : 1.f / 448.f;
    constexpr float one_over_six = 1.f / 6.f;
    float inv_val_max = scale_override * one_over_six;
    float scale = global_abs_max == 0 ? 1.f : global_abs_max * inv_scales_max * inv_val_max;
    if (idx == 0) {
        global_scale_ptr[0] = scale;
    }

    // per-group abs-maxes
    nv_bfloat16 local_abs_max = vecReduceAbsMax(x);
    // shuffle with neighbour. Can't use __reduce_max_sync because that doesn't allow partial reductions
    nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
    float full_abs_max = static_cast<float>(__hmax(local_abs_max, other_abs_max));

    // six
    QuantResult results[NumCandidates];
    float scores[NumCandidates];
    float best = INFINITY;
    QuantResult res;
    for (int i = 0; i < NumCandidates; ++i) {
        float inv_val = get_candidate_helper<Candidates...>::get_inv(i);
        results[i] = quantize(full_abs_max, inv_val * scale_override, scale, x);
        scores[i] = quant_error(x, results[i], scale);
        if (scores[i] < best) {
            best = scores[i];
            res = results[i];
        }
    }
    res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * idx);
    if (idx % 2 == 0) {
        int col = (idx / 2) % cols;
        int row = (idx / 2) / cols;
        auto tgt = cvt_quant_to_fp4_get_sf_out_offset(row, col, cols / 4);
        scale_ptr[tgt] = res.fp8s;
    }
}

void four_six_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long rows, long cols) {
    if (cols % 128 != 0) throw std::runtime_error("four_six_fp4: cols must be divisible by 128");
    int n_vecs = rows * cols / 8;
    int block_size = 256;
    int n_blocks = (n_vecs + block_size - 1) / block_size;
    four_six_fp4_kernel<6.f, 4.f><<<n_blocks, block_size>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_ptr, scale_override, n_vecs, cols / 16);
    CUDA_CHECK(cudaGetLastError());
}

void rtn_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long rows, long cols) {
    if (cols % 128 != 0) throw std::runtime_error("rtn_fp4: cols must be divisible by 128");
    int n_vecs = (rows * cols) / 8;
    int block_size = 256;
    int n_blocks = (n_vecs + block_size - 1) / block_size;
    four_six_fp4_kernel<6.f><<<n_blocks, block_size>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_ptr, scale_override, n_vecs, cols / 16);
    CUDA_CHECK(cudaGetLastError());
}
