// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <cuda/cmath>
#include <cuda_bf16.h>
#include "utils.cuh"
#include "vec.cuh"

constexpr int NUM_WARPS = 12;

template<bool Transpose>
__global__ __launch_bounds__(NUM_WARPS*32, 1) void cutlass_group_transform_128_eden_kernel(__nv_fp4x2_storage_t* y, nv_bfloat16* scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x, int rows, int cols, float inv_fp4_max) {
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = NUM_WARPS;

    constexpr int T_PER_G = G / T;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = warp_id + blockIdx.x * W;

    __shared__ nv_bfloat16 h_smem[G * G];
    extern __shared__ uint4 dynamic_smem[];

    nv_bfloat16 local_scale_max = 0.f;
    nv_bfloat16* a_smem = reinterpret_cast<nv_bfloat16*>(dynamic_smem) + T*G*warp_id;
    for (int k = warp_id; k < T_PER_G * T_PER_G; k += W) {
        int i = k % T_PER_G;
        int j = k / T_PER_G;
        // upper-left corner of the tiles
        nv_bfloat16* smem_base = h_smem + i * T * T + j * T * G;
        const nv_bfloat16* gmem_base = h + i * T + j * T * G;
        global_to_shared_swizzle_H(smem_base, gmem_base, G);
    }
    __pipeline_commit();

    const int groups = rows * cols / (G * T);
    if (int i = start_i; i < groups) {
        if constexpr (Transpose) {
            const int col = (i * T) % cols;
            const int row = (i * T) / cols * G;
            for (int s = 0; s < T_PER_G; ++s) {
                nv_bfloat16* smem_base = a_smem + s * T * T;
                const nv_bfloat16* gmem_base = x + col + (row + s * T) * cols;
                global_to_shared_16_32_swizzle(smem_base, gmem_base, cols);
            }
        } else {
            for (int s = 0; s < T_PER_G; ++s) {
                nv_bfloat16* smem_base = a_smem + s * T * T;
                const nv_bfloat16* gmem_base = x + s * T + i * T * G;
                global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
            }
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    for (int i = start_i; i < groups; i += W * gridDim.x) {
        __pipeline_wait_prior(0);
        for (int k = 0; k < T_PER_G; ++k) {
            if constexpr (Transpose) {
                a_frags[k] = load_fragment_aT_swizzle(lane_id, a_smem + k * T * T);
            } else {
                a_frags[k] = load_fragment_a_swizzle(lane_id, a_smem + k * T * T);
            }
        }

        if (const int next = i + W * gridDim.x; next < groups) {
            if constexpr (Transpose) {
                const int col = (next * T) % cols;
                const int row = (next * T) / cols * G;
                for (int s = 0; s < T_PER_G; ++s) {
                    nv_bfloat16* smem_base = a_smem + s * T * T;
                    const nv_bfloat16* gmem_base = x + col + (row + s * T) * cols;
                    global_to_shared_16_32_swizzle(smem_base, gmem_base, cols);
                }
            } else {
                for (int s = 0; s < T_PER_G; ++s) {
                    nv_bfloat16* smem_base = a_smem + s * T * T;
                    const nv_bfloat16* gmem_base = x + s * T + next * T * G;
                    global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
                }
            }
        }
        __pipeline_commit();

        m16_n16_k32_c_fragment<float> acc[T_PER_G];
        for (int j = 0; j < T_PER_G; ++j) {
            for (int k = 0; k < T_PER_G; ++k) {
                const nv_bfloat16* smem_base = h_smem + k * T * T + j * T * G;
                m16_n16_b_fragment<nv_bfloat16> b_frag = load_fragment_b_swizzle(lane_id, smem_base);
                mma_m16_n16_sync(acc[j], a_frags[k], b_frag, acc[j]);
            }
        }

        // -------------------------------------------------------------------------------------------------------------
        // epilogue

        using group_f_vec = GenericVector<float, 4>;
        using group_n_vec = GenericVector<__nv_fp4x2_storage_t, 2>;
        constexpr int SPT = 16;      // scales per thread

        nv_bfloat16 out_scales[SPT];

        // reorganize data
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            #pragma unroll
            for (int j = 0; j < T_PER_G; ++j) {
                group_f_vec nv_group;
                nv_group[0] = acc[j].v[0 + 2*k];
                nv_group[1] = acc[j].v[1 + 2*k];
                nv_group[2] = acc[j].v[4 + 2*k];
                nv_group[3] = acc[j].v[5 + 2*k];
                float abs_max = fabsf(nv_group[0]);
                // determine local abs-max.
                for (int g = 1; g < group_f_vec::size; ++g) {
                    abs_max = fmaxf(abs_max, fabsf(nv_group[g]));
                }
                // now reduce over the quads that collectively hold the 16 elements:
                abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, 1));
                abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, 2));

                float scale = abs_max * inv_fp4_max;
                //                          SEEE EEEE EMMM M     19x0
                constexpr unsigned MASK = 0b1111'1111'1111'1u << 19u;
                // RTZ with 4 bits produces better results than RTN with 3 bits, but leaving all bits
                // is worse ?!
                // constexpr unsigned MASK = 0b1111'1111'1111u << 20u;
                float m3_scale = __uint_as_float((__float_as_uint(scale) & MASK));
                // TODO Masking == RTZ, is this OK here?

                float factor = m3_scale > 0 ? reciprocal_approximate_ftz(m3_scale) : 0.f;

                group_n_vec converted;
                float x_y = 0.f;
                float x_x = 0.f;

                for (int t = 0; t < group_f_vec::size; t += 2) {
                    float2 v = make_float2(nv_group[t] * factor, nv_group[t+1] * factor);
                    __nv_fp4x2_storage_t bits = __nv_cvt_float2_to_fp4x2(v, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
                    converted[t/2] = bits;
                    float2 back = __nv_cvt_fp4x2_to_float2(bits);
                    x_x += v.x * v.x + v.y * v.y;
                    x_y += v.x * back.x + v.y * back.y;
                }

                // reduce quad
                x_x += __shfl_xor_sync(0xFFFFFFFFu, x_x, 1);
                x_y += __shfl_xor_sync(0xFFFFFFFFu, x_y, 1);
                x_x += __shfl_xor_sync(0xFFFFFFFFu, x_x, 2);
                x_y += __shfl_xor_sync(0xFFFFFFFFu, x_y, 2);

                float correction = (x_y == 0) ? 1.f : x_x * reciprocal_approximate_ftz(x_y);
                float fixed_scale = m3_scale * correction;
                out_scales[j + T_PER_G*k] = static_cast<nv_bfloat16>(fixed_scale);

                int t4 = lane_id % 4;
                int r4 = lane_id / 4;
                if constexpr (Transpose) {
                    const int col = (i * T) % cols;
                    const int row = (i * T) / cols * G;

                    __nv_fp4x2_storage_t* y_base = y + (col + r4 + 8*k) * rows/2 + row/2;
                    converted.store(y_base + 2 * t4 + j*8);
                } else {
                    __nv_fp4x2_storage_t* y_base = y + (i * T + 8 * k) * G / 2 + r4 * G/2;
                     converted.store(y_base + 2 * t4 + j*8);
                }
            }
        }

        if (lane_id % 4 == 0) {
            using scales_vec = GenericVector<nv_bfloat16, 8>;
            scales_vec sv;
            int r4 = lane_id / 4;
            for (int r = 0; r < 2; ++r) {
                for (int k = 0; k < 8; ++k) {
                    sv[k] = out_scales[k + 8 * r];
                }
                if constexpr (Transpose) {
                    const int col = (i * T) % cols;
                    const int row = (i * T) / cols * G;
                    sv.store(scales + (col + r4 + r * 8) * rows / 16 + row / 16);
                } else {
                    sv.store(scales + (i * T + r4 + r * 8) * G / 16);
                }
                local_scale_max = fmaxf(local_scale_max, vecReduceAbsMax(sv));
            }
        }
    }

    // ok, finally, reduce the abs-max per warp. note that each quad already is in sync, so we only need cross-quad shuffles
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 4));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 8));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 16));

    if (lane_id == 0) {
        unsigned as_32_bits = __float_as_uint(static_cast<float>(local_scale_max));
        atomicMax(max_scale, as_32_bits);
    }
}

__global__ void eden_convert_scales_kernel(__nv_fp8_e4m3* scales_fp8, float* global_scale_ptr, const nv_bfloat16* scales_bf16, const unsigned* max_scale_ptr, long seed, float inv_fp8_max, int rows, int cols) {
    using bf16x8 = GenericVector<nv_bfloat16, 8>;
    using fp32x8 = GenericVector<float, 8>;
    using fp8x8 = GenericVector<__nv_fp8_e4m3, 8>;
    int col = 8*(threadIdx.x + blockIdx.x * blockDim.x);
    int row = threadIdx.y + blockIdx.y * 64;
    if (col >= cols || row >= rows) return;
    uint4 rng = philox<10>(seed, threadIdx.x, blockIdx.x);
    cudaGridDependencySynchronize();

    float max_scale = __uint_as_float(*max_scale_ptr);
    float global_scale = max_scale * inv_fp8_max;
    if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0) {
        global_scale_ptr[0] = global_scale;
    }
    float factor = global_scale != 0 ? reciprocal_approximate_ftz(global_scale) : 0.f;
    bf16x8 src_scales[2];
    src_scales[0] = bf16x8::load(scales_bf16 + col + row * cols);
    src_scales[1] = bf16x8::load(scales_bf16 + col + (row+32) * cols);
    fp8x8 dst_scales;
    fp32x8 pre_sr[2];
    for (int g = 0; g < 2; ++g) {
        for (int k = 0; k < 8; ++k) {
            float src_scale = static_cast<float>(src_scales[g][k]);
            pre_sr[g][k] = src_scale * factor;
        }
    }

    auto lo = [](unsigned int a) -> unsigned short {
        return a & 0xffff;
    };
    auto hi = [](unsigned int a) -> unsigned short {
        return (a >> 16) & 0xffff;
    };

    dst_scales[0] = stochastic_rounding(pre_sr[0][0], lo(rng.x));
    dst_scales[1] = stochastic_rounding(pre_sr[0][1], lo(rng.y));
    dst_scales[2] = stochastic_rounding(pre_sr[0][2], lo(rng.z));
    dst_scales[3] = stochastic_rounding(pre_sr[0][3], lo(rng.w));
    dst_scales[4] = stochastic_rounding(pre_sr[1][0], hi(rng.x));
    dst_scales[5] = stochastic_rounding(pre_sr[1][1], hi(rng.y));
    dst_scales[6] = stochastic_rounding(pre_sr[1][2], hi(rng.z));
    dst_scales[7] = stochastic_rounding(pre_sr[1][3], hi(rng.w));
    // with sf swizzling, row+32 corresponds to elements 4-7, so this works out
    dst_scales.store(scales_fp8 + cvt_quant_to_fp4_get_sf_out_offset(row, col, cols / 4));

    rng = philox<10>(seed + 1, threadIdx.x, blockIdx.x);
    dst_scales[0] = stochastic_rounding(pre_sr[0][4], lo(rng.x));
    dst_scales[1] = stochastic_rounding(pre_sr[0][5], lo(rng.y));
    dst_scales[2] = stochastic_rounding(pre_sr[0][6], lo(rng.z));
    dst_scales[3] = stochastic_rounding(pre_sr[0][7], lo(rng.w));
    dst_scales[4] = stochastic_rounding(pre_sr[1][4], hi(rng.x));
    dst_scales[5] = stochastic_rounding(pre_sr[1][5], hi(rng.y));
    dst_scales[6] = stochastic_rounding(pre_sr[1][6], hi(rng.z));
    dst_scales[7] = stochastic_rounding(pre_sr[1][7], hi(rng.w));
    dst_scales.store(scales_fp8 + cvt_quant_to_fp4_get_sf_out_offset(row, col + 4, cols / 4));
}

void launch_eden_convert_scales_kernel(__nv_fp8_e4m3* scales_fp8, float* global_scale_ptr, const nv_bfloat16* scales_bf16, const unsigned* max_scale_ptr, long seed, float inv_fp8_max, int rows, int cols) {
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config;
    config.attrs = attribute;
    config.numAttrs = 1;
    config.blockDim = dim3(8, 32, 1);
    config.gridDim = dim3(cuda::ceil_div(cols, 8*8), rows / 64, 1);
    config.dynamicSmemBytes = 0;
    config.stream = nullptr;
    CUDA_CHECK(cudaLaunchKernelEx(&config, eden_convert_scales_kernel, scales_fp8, global_scale_ptr, scales_bf16, max_scale_ptr, seed, inv_fp8_max, rows, cols));
}

template<bool TransposeA>
void group_transform_128_eden_launcher(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N)
{
    if (N % 128 != 0) throw std::runtime_error("group_transform_128: N must be divisible by 128");
    if (M % 128 != 0) throw std::runtime_error("group_transform_128: M must be divisible by 128");

    int blocks, device;
    int smem = NUM_WARPS * 16 * 128 * 2;
    CUDA_CHECK(cudaGetDevice(&device));

    CUDA_CHECK(cudaFuncSetAttribute(cutlass_group_transform_128_eden_kernel<TransposeA>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, cutlass_group_transform_128_eden_kernel<TransposeA>, 32*NUM_WARPS, smem));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

    CUDA_CHECK(cudaMemset(max_scale, 0, sizeof(unsigned)));

    cutlass_group_transform_128_eden_kernel<TransposeA><<<sms * blocks, dim3(32, NUM_WARPS), smem>>>(y, scratch_scales, max_scale, h, x, M, N, 1.f / fp4_max);
    CUDA_CHECK(cudaGetLastError());
    launch_eden_convert_scales_kernel(scales_fp8, global_scale_ptr, scratch_scales, max_scale, seed, 1.f / fp8_max, TransposeA ? N : M, TransposeA ? M / 16 : N / 16);
}

void group_transform_128_eden(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N, bool transposeX)
{
    if (transposeX) {
        group_transform_128_eden_launcher<true>(y, scales_fp8, global_scale_ptr, scratch_scales, max_scale, h, x, seed, fp4_max, fp8_max, M, N);
    } else {
        group_transform_128_eden_launcher<false>(y, scales_fp8, global_scale_ptr, scratch_scales, max_scale, h, x, seed, fp4_max, fp8_max, M, N);
    }
}
