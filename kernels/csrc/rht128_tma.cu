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
__global__ void rht128_tma_kernel(nv_bfloat16* y, const nv_bfloat16* h_packed, const __grid_constant__ CUtensorMap x_tmap, int rows, int cols) {
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = 8;

    constexpr int T_PER_G = G / T;
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    namespace ptx = cuda::ptx;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = warp_id + blockIdx.x * W;

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
        // upper-left corner of the tiles
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

        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            for (int g = 0; g < 2; ++g) {
                using group_vec = GenericVector<nv_bfloat16, 16>;
                group_vec nv_group;
                for (int j = 0; j < T_PER_G / 2; ++j) {
                    nv_group[4*j+0] = static_cast<nv_bfloat16>(tile_prods[j + 4*g].v[0 + 2*k]);
                    nv_group[4*j+1] = static_cast<nv_bfloat16>(tile_prods[j + 4*g].v[1 + 2*k]);
                    nv_group[4*j+2] = static_cast<nv_bfloat16>(tile_prods[j + 4*g].v[4 + 2*k]);
                    nv_group[4*j+3] = static_cast<nv_bfloat16>(tile_prods[j + 4*g].v[5 + 2*k]);
                }

                int t4 = lane_id % 4;
                int r4 = lane_id / 4;
                if constexpr (Transpose) {
                    const int col = (i * T) % cols;
                    const int row = (i * T) / cols * G;
                    nv_bfloat16* y_base = y + (col + r4  + 8*k) * rows + row;
                    nv_group.store(y_base + 16 * t4 + 64 * g);
                } else {
                    nv_bfloat16* y_base = y + (i * T + 8*k) * G + r4 * G;
                    nv_group.store(y_base + 16 * t4 + 64 * g);
                }
            }
        }
    }
}

template<bool Transpose>
void rht128_tma_launcher(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N) {
    if (N % 128 != 0) throw std::runtime_error("rht128_tma: N must be divisible by 128");
    if (M % 128 != 0) throw std::runtime_error("rht128_tma: M must be divisible by 128");

    int blocks, device;
    int smem = 8 * 16 * 128 * 2 * sizeof(nv_bfloat16);
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaFuncSetAttribute(rht128_tma_kernel<Transpose>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rht128_tma_kernel<Transpose>, 32*8, smem));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    auto tmap = Transpose ? create_xT_tensor_map(x, M, N) : create_x_tensor_map(x, M, N);
    rht128_tma_kernel<Transpose><<<sms * blocks, dim3(32, 8), smem>>>(y, H, tmap, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void transform_rht128_tma(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose) {
    if (transpose) rht128_tma_launcher<true>(y, H, x, M, N);
    else rht128_tma_launcher<false>(y, H, x, M, N);
}

