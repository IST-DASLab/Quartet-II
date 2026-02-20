// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include "had128_utils.cuh"

//! \brief transform with sign-randomized hadamard matrix of size 128.
//! \details We assume Kronecker-structured hadamard. After sign-randomization,
//! it will look like this (in 16x16 tiles):
//! A  B  C  D   E  F  G  H
//! A -B  C -D   E -F  G -H
//! A  B -C -D   E  F -G -H
//! A -B -C  D   E -F -G  H
//! A  B  C  D  -E -F -G -H
//! A -B  C -D  -E  F -G  H
//! A  B -C -D  -E -F  G  H
//! A -B -C  D  -E  F  G -H
//! Therefore, this kernel only takes A-H as input, (8 16x16 tiles)
template<bool Transpose>
__global__ __launch_bounds__(32*8, 2) void rht128_eden_tma_kernel(__nv_fp4x2_storage_t* y, nv_bfloat16* scales, unsigned* max_scale, const nv_bfloat16* h_packed, const __grid_constant__ CUtensorMap x_tmap, int rows, int cols, float inv_fp4_max) {
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = 8;

    constexpr int T_PER_G = G / T;
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    namespace ptx = cuda::ptx;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = warp_id + blockIdx.x * W;

    nv_bfloat16 local_scale_max = 0.f;

    __shared__ alignas(16) nv_bfloat16 h_smem[G * T];
    __shared__ barrier bar[2*W];
    if (threadIdx.x < 2*W && threadIdx.y == 0) {
        init(bar + threadIdx.x, 32);
    }
    barrier* my_barrier = bar + 2*warp_id;
    __syncthreads();

    extern __shared__ AlignedSmem tma_dynamic_smem[];
    nv_bfloat16* const smem = reinterpret_cast<nv_bfloat16*>(tma_dynamic_smem);
    nv_bfloat16* a_smem = smem + T*G*warp_id;
    {
        nv_bfloat16* smem_base = h_smem + warp_id * T * T;
        const nv_bfloat16* gmem_base = h_packed + warp_id * T;
        global_to_shared_swizzle_H(smem_base, gmem_base, G);
    }
    __pipeline_commit();
    int phase = 0;
    const int groups = rows * cols / (G * T);

    auto ldgsts = [&](int i, int phase) {
        if (threadIdx.x == 0) {
            if constexpr (Transpose) {
                const int col = (i * T) % cols;
                const int row = (i * T) / cols * G;
                // Initiate bulk tensor copy.
                int32_t tensor_coords[2] = { col, row };
                ptx::cp_async_bulk_tensor(
    #if defined(__NVCC__) and (__CUDACC_VER_MAJOR__ >= 13 or (__CUDACC_VER_MAJOR__ == 12 and __CUDACC_VER_MINOR__ >= 9))
                  ptx::space_shared,
    #else
                  ptx::space_cluster,
    #endif
                  ptx::space_global,
                  a_smem + phase * T*G*W, &x_tmap, tensor_coords,
                  cuda::device::barrier_native_handle(my_barrier[phase])
                  );
            } else {
                // Initiate bulk tensor copy.
                int32_t tensor_coords[3] = { 0, 16*i,  0};
                ptx::cp_async_bulk_tensor(
    #if defined(__NVCC__) and (__CUDACC_VER_MAJOR__ >= 13 or (__CUDACC_VER_MAJOR__ == 12 and __CUDACC_VER_MINOR__ >= 9))
                  ptx::space_shared,
    #else
                  ptx::space_cluster,
    #endif
                  ptx::space_global,
                  a_smem + phase * T*G*W, &x_tmap, tensor_coords,
                  cuda::device::barrier_native_handle(my_barrier[phase])
                  );
            }

            // Arrive on the barrier and tell how many bytes are expected to come in.
            return cuda::device::barrier_arrive_tx(my_barrier[phase], 1, 128*16*sizeof(nv_bfloat16));
        } else {
            // Other threads just arrive.
            return my_barrier[phase].arrive();
        }
    };

    auto wait_token = ldgsts(start_i, 0);
    auto next_token = ldgsts(start_i + W * gridDim.x, 1);
    __pipeline_wait_prior(0);   //  wait for H to be loaded
    __syncthreads();

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    m16_n16_b_fragment<nv_bfloat16> b_frags[T_PER_G];
    // load RHT fragments into registers
    for (int k = 0; k < T_PER_G; ++k) {
        b_frags[k] = load_fragment_b_swizzle(lane_id, h_smem + k * T * T);
    }

    for (int i = start_i; i < groups; i += W * gridDim.x) {
        my_barrier[phase].wait(std::move(wait_token));

        ldsm_128x16<Transpose>(a_frags, a_smem + phase * T*G*W);

        wait_token = std::move(next_token);
        next_token = ldgsts(i + 2 * W * gridDim.x, phase);
        phase = (phase + 1) % 2;

        // now generate all tile-wise products
        m16_n16_k32_c_fragment<float> tile_prods[T_PER_G];
        had128_mma(a_frags, b_frags, tile_prods);
        eden_rounding<Transpose>(y, scales, local_scale_max, tile_prods, i, rows, cols, inv_fp4_max);
    }

    // ok, finally, reduce the abs-max per warp. note that each quad already is in sync, so we only need cross-quad shuffles
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 1));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 2));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 4));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 8));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 16));

    if (lane_id == 0) {
        unsigned as_32_bits = __float_as_uint(static_cast<float>(local_scale_max));
        atomicMax(max_scale, as_32_bits);
    }
}

void launch_eden_convert_scales_kernel(__nv_fp8_e4m3* scales_fp8, float* global_scale_ptr, const nv_bfloat16* scales_bf16, const unsigned* max_scale_ptr, long seed, float inv_fp8_max, int rows, int cols);

template<bool Transpose>
void rht128_eden_launcher(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N)
{
    if (N % 128 != 0) throw std::runtime_error("rht128_tma: N must be divisible by 128");
    if (M % 128 != 0) throw std::runtime_error("rht128_tma: M must be divisible by 128");

    int blocks, device;
    int smem = 8 * 16 * 128 * 2 * sizeof(nv_bfloat16);
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaFuncSetAttribute(rht128_eden_tma_kernel<Transpose>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rht128_eden_tma_kernel<Transpose>, 32*8, smem));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaMemset(max_scale, 0, sizeof(unsigned)));
    auto tmap = Transpose ? create_xT_tensor_map(x, M, N) : create_x_tensor_map(x, M, N);
    rht128_eden_tma_kernel<Transpose><<<sms * blocks, dim3(32, 8), smem>>>(y, scratch_scales, max_scale, h, tmap, M, N, 1.f / fp4_max);
    CUDA_CHECK(cudaGetLastError());
    launch_eden_convert_scales_kernel(scales_fp8, global_scale_ptr, scratch_scales, max_scale, seed, 1.f / fp8_max, Transpose ? N : M, Transpose ? M / 16 : N / 16);
}

void transform_rht128_eden(__nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N, bool transpose)
{
    if (transpose) rht128_eden_launcher<true>(y, scales_fp8, global_scale_ptr, scratch_scales, max_scale, h, x, seed, fp4_max, fp8_max, M, N);
    else rht128_eden_launcher<false>(y, scales_fp8, global_scale_ptr, scratch_scales, max_scale, h, x, seed, fp4_max, fp8_max, M, N);
}
