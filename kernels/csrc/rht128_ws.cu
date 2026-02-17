// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_bf16.h>
#include "had128_utils.cuh"

__device__ void rht128_producer( nv_bfloat16* x_smem, __mbarrier_t ready[], __mbarrier_t filled[], const nv_bfloat16* x, int groups) {
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = 8;

    constexpr int T_PER_G = G / T;

    int warp_id = threadIdx.y - W;      // 8 consumer warps
    int block_i = blockIdx.x * W;
    int phase = 0;

    x_smem = x_smem + 2 * T*G*warp_id;

    int col = threadIdx.x % 2;
    int row = threadIdx.x / 2;
    nv_bfloat16* smem_base = x_smem + swizzle_smem(row, col);
    const nv_bfloat16* gmem_base = x + col * 8 + row * G;

    // we use the same condition in every block; this kernel can only be called with groups%8 == 0, so that is fine
    for (int i = block_i; i < groups; i += gridDim.x * W) {
        __mbarrier_token_t token = __mbarrier_arrive(ready + phase); /* wait for buffer_(i%2) to be ready to be filled */
        while(!__mbarrier_try_wait(ready + phase, token, 1000)) {}

        for (int j = 0; j < 2; ++j) {
            int idx = i + j + 2 * warp_id;
            for (int s = 0; s < T_PER_G; ++s) {
                __pipeline_memcpy_async(
                            smem_base + s * T * T + phase * T*G*W + j * T * G,
                            gmem_base + s * T + idx * T * G,
                            16);
            }
        }
        __pipeline_arrive_on(filled + phase);
        __mbarrier_arrive(filled + phase);  /* buffer_(i%2) is filled */
        phase = (phase + 1) % 2;
    }
}

template<int Offset>
__device__ void ldsm_with_offset(uint4& v, size_t src ) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4+%5];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(src), "n"(Offset * sizeof(nv_bfloat16))
               );
}

__device__ void rht128_consumer( nv_bfloat16* y, const nv_bfloat16* h_smem, const nv_bfloat16* x_smem, __mbarrier_t ready[], __mbarrier_t filled[], int groups) {
    __mbarrier_arrive(ready + 0);
    __mbarrier_arrive(ready + 1);

    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = 8;

    constexpr int T_PER_G = G / T;

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int block_i = blockIdx.x * W;
    int phase = 0;
    int t8 = lane_id % 8;
    int g8 = lane_id / 8;
    x_smem = x_smem + T*G*warp_id + swizzle_smem(t8 + 8 * (g8%2), g8 / 2);
    size_t x_smem_shared = __cvta_generic_to_shared(x_smem);

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    m16_n16_b_fragment<nv_bfloat16> b_frags[T_PER_G];
    // load RHT fragments into registers
    for (int k = 0; k < T_PER_G; ++k) {
        b_frags[k] = load_fragment_b_swizzle(lane_id, h_smem + k * T * T);
    }

    // we use the same condition in every block; this kernel can only be called with groups%8 == 0, so that is fine
    for (int i = block_i; i < groups; i += gridDim.x * W) {
        __mbarrier_token_t token = __mbarrier_arrive(filled + phase);
        while(!__mbarrier_try_wait(filled + phase, token, 1000)) {}
        const size_t  src = x_smem_shared + phase * T*G*W;
        ldsm_with_offset<0*T*T>(a_frags[0].v, src);
        ldsm_with_offset<1*T*T>(a_frags[1].v, src);
        ldsm_with_offset<2*T*T>(a_frags[2].v, src);
        ldsm_with_offset<3*T*T>(a_frags[3].v, src);
        ldsm_with_offset<4*T*T>(a_frags[4].v, src);
        ldsm_with_offset<5*T*T>(a_frags[5].v, src);
        ldsm_with_offset<6*T*T>(a_frags[6].v, src);
        ldsm_with_offset<7*T*T>(a_frags[7].v, src);
        __mbarrier_arrive(ready + phase); /* buffer_(i%2) is ready to be re-filled */
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
                nv_bfloat16* y_base = y + (i * T + 8*k) * G + r4 * G;
                nv_group.store(y_base + 16 * t4 + 64 * g);
            }
        }
    }
}

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
__global__ void __maxnreg__(120) rht128_ws_kernel(nv_bfloat16* y, const nv_bfloat16* h_packed, const nv_bfloat16* x, int groups) {
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = 8;

    __shared__ alignas(16) nv_bfloat16 h_smem[G * T];
    __shared__ __mbarrier_t barriers[4];
    extern __shared__ uint4 dynamic_smem[];
    int warp_id = threadIdx.y;

    if (warp_id == 0 && threadIdx.x < 4) {
        __mbarrier_init(barriers + threadIdx.x, (W + 4) * 32);
    }

    // at this point, the consumer warps don't actually need this many registers
    /*if (warp_id < 8) {
        asm ("setmaxnreg.inc.sync.aligned.u32 120;");
    } else {
        asm ("setmaxnreg.dec.sync.aligned.u32 40;");
    }*/
    if (warp_id < 8) {
        // upper-left corner of the tiles
        nv_bfloat16* smem_base = h_smem + warp_id * T * T;
        const nv_bfloat16* gmem_base = h_packed + warp_id * T;
        global_to_shared_swizzle_H(smem_base, gmem_base, G);
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
    __syncthreads();

    if (warp_id < 8) {
        rht128_consumer(y, h_smem, reinterpret_cast<nv_bfloat16*>(dynamic_smem), barriers, barriers + 2, groups);
    } else {
        rht128_producer(reinterpret_cast<nv_bfloat16*>(dynamic_smem), barriers, barriers + 2, x, groups);
    }
}


void rht128_ws_launcher(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N) {
    assert(M * N % 128 == 0);
    int groups = M * N / 128;
    int blocks, device;
    int smem = 8 * 16 * 128 * 4;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaFuncSetAttribute(rht128_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rht128_ws_kernel, 32*12, smem));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    rht128_ws_kernel<<<sms * blocks, dim3(32, 12), smem>>>(y, H, x, groups / 16);
    CUDA_CHECK(cudaGetLastError());
}

void transform_rht128_ws(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose) {
    if (transpose) throw std::runtime_error("transpose not supported for RHT128");
    else rht128_ws_launcher(y, H, x, M, N);
}
