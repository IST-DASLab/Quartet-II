// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef EDEN_HAD128_UTILS_CUH
#define EDEN_HAD128_UTILS_CUH


#include <cmath>

#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

#include "vec.cuh"
#include "utils.cuh"


inline CUtensorMap create_x_tensor_map(const nv_bfloat16* src, int M, int N) {
    CUtensorMap tensor_map{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 3;
    uint64_t size[rank] = {64, M*N/128u, 2};
    // The stride is the number of bytes to traverse from the first element of one
    // row to the next. It must be a multiple of 16.
    uint64_t stride[rank - 1] = {128 * sizeof(nv_bfloat16), 64 * sizeof(nv_bfloat16)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {64, 16, 2};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for
    // instance.
    uint32_t elem_stride[rank] = {1, 1, 1};

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
      &tensor_map,  // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,         // cuuint32_t tensorRank,
      (void*)src,   // void *globalAddress,
      size,         // const cuuint64_t *globalDim,
      stride,       // const cuuint64_t *globalStrides,
      box_size,     // const cuuint32_t *boxDim,
      elem_stride,  // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      // L2 Promotion can be used to widen the effect of a cache-policy to a
      // wider set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // Any element that is outside of bounds will be set to zero by the TMA
      // transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to create x tensor map");
    }

    return tensor_map;
}

inline CUtensorMap create_xT_tensor_map(const nv_bfloat16* src, int M, int N) {
    CUtensorMap tensor_map{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {(unsigned)N, (unsigned)M};
    // The stride is the number of bytes to traverse from the first element of one
    // row to the next. It must be a multiple of 16.
    uint64_t stride[rank - 1] = {N * sizeof(nv_bfloat16)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {16, 128};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for
    // instance.
    uint32_t elem_stride[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
      &tensor_map,  // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,         // cuuint32_t tensorRank,
      (void*)src,   // void *globalAddress,
      size,         // const cuuint64_t *globalDim,
      stride,       // const cuuint64_t *globalStrides,
      box_size,     // const cuuint32_t *boxDim,
      elem_stride,  // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
      // L2 Promotion can be used to widen the effect of a cache-policy to a
      // wider set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // Any element that is outside of bounds will be set to zero by the TMA
      // transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to create x tensor map");
    }

    return tensor_map;
}


inline CUtensorMap create_xT_fp4_tensor_map(const __nv_fp4x2_storage_t* src, int M, int N) {
    CUtensorMap tensor_map{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {(unsigned)N, (unsigned)M};
    // The stride is the number of bytes to traverse from the first element of one
    // row to the next. It must be a multiple of 16.
    uint64_t stride[rank - 1] = {N / 2u};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {32, 128};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for
    // instance.
    uint32_t elem_stride[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
      &tensor_map,  // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
      rank,         // cuuint32_t tensorRank,
      (void*)src,   // void *globalAddress,
      size,         // const cuuint64_t *globalDim,
      stride,       // const cuuint64_t *globalStrides,
      box_size,     // const cuuint32_t *boxDim,
      elem_stride,  // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
      // L2 Promotion can be used to widen the effect of a cache-policy to a
      // wider set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // Any element that is outside of bounds will be set to zero by the TMA
      // transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to create x tensor map");
    }

    return tensor_map;
}

template<bool Transpose>
__device__ __forceinline__ void eden_rounding(
    __nv_fp4x2_storage_t* y, nv_bfloat16* scales, nv_bfloat16& local_scale_max,
    const m16_n16_k32_c_fragment<float> (&tile_prods)[8],
    int i, int rows, int cols, float inv_fp4_max)
{
    constexpr int T = 16;
    constexpr int G = 128;

    using group_f_vec = GenericVector<float, 16>;
    using group_n_vec = GenericVector<__nv_fp4x2_storage_t, 8>;

    int lane_id = threadIdx.x;

    // reorganize data
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        for (int g = 0; g < 2; ++g) {
            group_f_vec nv_group;
            for (int j = 0; j < 4; ++j) {
                nv_group[4*j+0] = tile_prods[j + 4*g].v[0 + 2*k];
                nv_group[4*j+1] = tile_prods[j + 4*g].v[1 + 2*k];
                nv_group[4*j+2] = tile_prods[j + 4*g].v[4 + 2*k];
                nv_group[4*j+3] = tile_prods[j + 4*g].v[5 + 2*k];
            }

            float abs_max = fabsf(nv_group[0]);
            for (int j = 1; j < group_f_vec::size; ++j) {
                abs_max = fmaxf(abs_max, fabsf(nv_group[j]));
            }
            float scale = abs_max * inv_fp4_max;

            //                          SEEE EEEE EMMM M     19x0
            constexpr unsigned MASK = 0b1111'1111'1111'1u << 19u;
            // RTZ with 4 bits produces better results than RTN with 3 bits, but leaving all bits
            // is worse ?!
            // constexpr unsigned MASK = 0b1111'1111'1111u << 20u;
            float m3_scale = __uint_as_float((__float_as_uint(scale) & MASK));
            float factor = m3_scale > 0 ? reciprocal_approximate_ftz(m3_scale) : 0.f;

            group_n_vec converted;
            float2 x_y = {0.f, 0.f};
            float2 x_x = {0.f, 0.f};

            for (int t = 0; t < group_f_vec::size; t += 2) {
                float2 v = make_float2(nv_group[t] * factor, nv_group[t+1] * factor);
                __nv_fp4x2_storage_t bits = __nv_cvt_float2_to_fp4x2(v, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
                converted[t/2] = bits;
                float2 back = __nv_cvt_fp4x2_to_float2(bits);
                x_x = __ffma2_rn(v, v, x_x);
                x_y = __ffma2_rn(v, back, x_y);
            }

            float xx_t = x_x.x + x_x.y;
            float xy_t = x_y.x + x_y.y;
            float correction = (xy_t == 0) ? 1.f : xx_t * reciprocal_approximate_ftz(xy_t);
            float fixed_scale = m3_scale * correction;

            int t4 = lane_id % 4;
            int r4 = lane_id / 4;
            if constexpr (Transpose) {
                const int col = (i * T) % cols;
                const int row = (i * T) / cols * G;
                __nv_fp4x2_storage_t* y_base = y + (col + r4  + 8 * k) * rows/2 + row/2;
                converted.store(y_base + 8 * t4 + 32 * g);
                scales[(col + r4 + k * 8) * rows / 16 + row / 16 + t4 + 4 * g] = static_cast<nv_bfloat16>(fixed_scale);
            } else {
                __nv_fp4x2_storage_t* y_base = y + (i * T + 8*k + r4) * G / 2;
                converted.store(y_base + 8 * t4 + 32 * g);
                scales[(i * T + r4 + k * 8) * G / 16 + t4 + 4 * g] = static_cast<nv_bfloat16>(fixed_scale);
            }
            local_scale_max = fmaxf(local_scale_max, fixed_scale);
        }
    }
}

__device__ __forceinline__ void had128_mma(
    const m16_n16_a_fragment<nv_bfloat16> (&a_frags)[8],
    const m16_n16_b_fragment<nv_bfloat16> (&b_frags)[8],
    m16_n16_k32_c_fragment<float> (&tile_prods)[8])
{
    constexpr int T_PER_G = 8;
    for (int k = 0; k < T_PER_G; ++k) {
        mma_m16_n16_sync(tile_prods[k], a_frags[k], b_frags[k], tile_prods[k]);
    }
    // now generate linear combinations of 2 elements
    for (int k = 0; k < T_PER_G; k += 2) {
        auto plus = add_c_fragments(tile_prods[k], tile_prods[k+1]);
        auto minus = sub_c_fragments(tile_prods[k], tile_prods[k+1]);
        tile_prods[k] = plus;
        tile_prods[k+1] = minus;
    }

    // linear combinations of 4 elements
    // [0] A  B  C  D
    // [1] A -B  C -D
    // [2] A  B -C -D
    // [3] A -B -C  D
    for (int k = 0; k < 2; ++k) {
        for (int s = 0; s < T_PER_G; s += 4) {
            auto plus = add_c_fragments(tile_prods[k+s+0], tile_prods[k+s+2]);
            auto minus = sub_c_fragments(tile_prods[k+s+0], tile_prods[k+s+2]);
            tile_prods[k+s+0] = plus;
            tile_prods[k+s+2] = minus;
        }
    }

    for (int k = 0; k < 4; ++k) {
        auto plus = add_c_fragments(tile_prods[k], tile_prods[k+4]);
        auto minus = sub_c_fragments(tile_prods[k+0], tile_prods[k+4]);
        tile_prods[k] = plus;
        tile_prods[k+4] = minus;
    }
}

template<bool Transpose>
__device__ __forceinline__ void ldsm_128x16(m16_n16_a_fragment<nv_bfloat16> (&a_frags)[8], const nv_bfloat16* smem) {
    int lane_id = threadIdx.x;
    int t8 = lane_id % 8;
    int g8 = lane_id / 8;

    auto load_fragment  = [t8, g8](const nv_bfloat16* base, int k) {
        m16_n16_a_fragment<nv_bfloat16> result;

        const nv_bfloat16* src;
        if constexpr (Transpose) {
            int row = t8 + 8 * (g8/2) + 16*k;
            int col = (g8 % 2);
            int rc = col ^ ((row / 4) % 2);
            src = base + row * 16 + rc * 8;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
           : "=r"(result.v.x), "=r"(result.v.y), "=r"(result.v.z), "=r"(result.v.w)
           : "l"(__cvta_generic_to_shared(src))
           );

        } else {
            int row = t8 + 8 * (g8%2);
            int col = g8 / 2 + 2*k;
            int rc = (col % 8) ^ (row % 8);
            src = base + 64 * row + 8 * rc + 64*16*(col/8);

            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
           : "=r"(result.v.x), "=r"(result.v.y), "=r"(result.v.z), "=r"(result.v.w)
           : "l"(__cvta_generic_to_shared(src))
           );
        }

        return result;
    };

    for (int k = 0; k < 8; ++k) {
        a_frags[k] = load_fragment(smem, k);
    }
}

#endif //EDEN_HAD128_UTILS_CUH