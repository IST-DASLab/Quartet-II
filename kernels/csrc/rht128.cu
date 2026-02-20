// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_bf16.h>
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
__global__ void rht128_kernel(nv_bfloat16* y, const nv_bfloat16* h_packed, const nv_bfloat16* x, int rows, int cols) {
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = 8;

    constexpr int T_PER_G = G / T;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = warp_id + blockIdx.x * W;

    __shared__ alignas(16) nv_bfloat16 h_smem[G * T];
    extern __shared__ uint4 dynamic_smem[];
    nv_bfloat16* a_smem = reinterpret_cast<nv_bfloat16*>(dynamic_smem) + T*G*warp_id;
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
        if (i < groups) {
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
                    nv_bfloat16* smem_base = a_smem + s * T * T + phase * T*G*W;
                    const nv_bfloat16* gmem_base = x + s * T + i * T * G;
                    global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
                }
            }
        }
        __pipeline_commit();
    };

    ldgsts(start_i, 0);
    ldgsts(start_i + W * gridDim.x, 1);
    __pipeline_wait_prior(2);
    __syncthreads();

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    m16_n16_b_fragment<nv_bfloat16> b_frags[T_PER_G];
    // load RHT fragments into registers
    for (int k = 0; k < T_PER_G; ++k) {
        b_frags[k] = load_fragment_b_swizzle(lane_id, h_smem + k * T * T);
    }

    for (int i = start_i; i < groups; i += W * gridDim.x) {

        __pipeline_wait_prior(1);
        for (int k = 0; k < T_PER_G; ++k) {
            if constexpr (Transpose) {
                a_frags[k] = load_fragment_aT_swizzle(lane_id, a_smem + k * T * T + phase * T*G*W);
            } else {
                a_frags[k] = load_fragment_a_swizzle(lane_id, a_smem + k * T * T + phase * T*G*W);
            }
        }
        ldgsts(i + 2 * W * gridDim.x,  phase);
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
void rht128_launcher(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N) {
    if (N % 128 != 0) throw std::runtime_error("rht128: N must be divisible by 128");
    if (M % 128 != 0) throw std::runtime_error("rht128: M must be divisible by 128");

    int blocks, device;
    int smem = 8 * 16 * 128 * 4;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaFuncSetAttribute(rht128_kernel<Transpose>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rht128_kernel<Transpose>, 32*8, smem));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    rht128_kernel<Transpose><<<sms * blocks, dim3(32, 8), smem>>>(y, H, x, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void transform_rht128(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose) {
    if (transpose) rht128_launcher<true>(y, H, x, M, N);
    else rht128_launcher<false>(y, H, x, M, N);
}

