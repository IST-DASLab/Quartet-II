// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda/barrier>

#include "had128_utils.cuh"
#include "strided_iter.cuh"

constexpr int NUM_WARPS = 8;

// TODO find a way to unify with the other implementation
__device__ __forceinline__ void eden_rounding_tp(
    __nv_fp4x2_storage_t* y, nv_bfloat16* scales, nv_bfloat16& local_scale_max,
    const m16_n16_k32_c_fragment<float> (&tile_prods)[8],
    int row, int col, int rows, int cols, float inv_fp4_max)
{
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
            __nv_fp4x2_storage_t* y_base = y + (col + r4  + 8 * k) * rows/2 + row/2;
            converted.store(y_base + 8 * t4 + 32 * g);
            scales[(col + r4 + k * 8) * rows / 16 + row / 16 + t4 + 4 * g] = static_cast<nv_bfloat16>(fixed_scale);
            local_scale_max = fmaxf(local_scale_max, fixed_scale);
        }
    }
}

__global__ __launch_bounds__(NUM_WARPS*32, 1) void rht128_requant_kernel(
    __nv_fp4x2_storage_t* y, nv_bfloat16* scales, unsigned* max_scale, const nv_bfloat16* h,
    const __grid_constant__ CUtensorMap x_tmap, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    int rows, int cols, float inv_fp4_max)
{
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = NUM_WARPS;
    constexpr int WarpSize = 32;

    constexpr int T_PER_G = G / T;

    using barrier = cuda::barrier<cuda::thread_scope_block>;
    namespace ptx = cuda::ptx;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = 4*(warp_id + blockIdx.x * W);
    StridedIterator<int> iter(start_i * T, 4*W * gridDim.x * T, cols);

    __shared__ nv_bfloat16 h_smem[G * T];
    extern __shared__ AlignedSmem dynamic_smem[];
    __shared__ barrier bar[2*W];
    if (threadIdx.x < 2*W && threadIdx.y == 0) {
        init(bar + threadIdx.x, 32);
    }
    barrier* my_barrier = bar + warp_id;
    __syncthreads();

    nv_bfloat16 local_scale_max = 0.f;
    __nv_fp4x2_storage_t* smem_as_fp4 = reinterpret_cast<__nv_fp4x2_storage_t*>(dynamic_smem);
    __nv_fp4x2_storage_t* a_q_smem = smem_as_fp4 + 2*T*G*warp_id;
    nv_bfloat16* a_smem = reinterpret_cast<nv_bfloat16*>(smem_as_fp4 + 2 * T * G * NUM_WARPS) + T*G*warp_id;
    __nv_fp8_e4m3* scales_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(smem_as_fp4 + 4 * T * G * NUM_WARPS) + 4*G*warp_id;
    {
        nv_bfloat16* smem_base = h_smem + warp_id * T * T;
        const nv_bfloat16* gmem_base = h + warp_id * T;
        global_to_shared_swizzle_H(smem_base, gmem_base, G);
    }
    __pipeline_commit();

    float global_scale = *x_global_scale;

    auto ldgsts = [&]() {
        const int col = iter.get<1>();
        const int row = iter.get<0>() * G;

        barrier::arrival_token token;
        if (threadIdx.x == 0) {
            // Initiate bulk tensor copy.
            int32_t tensor_coords[2] = { col / 2, row };
            ptx::cp_async_bulk_tensor(
#if defined(__NVCC__) and (__CUDACC_VER_MAJOR__ >= 13 or (__CUDACC_VER_MAJOR__ == 12 and __CUDACC_VER_MINOR__ >= 9))
              ptx::space_shared,
#else
              ptx::space_cluster,
#endif
              ptx::space_global,
              a_q_smem, &x_tmap, tensor_coords,
              cuda::device::barrier_native_handle(*my_barrier)
              );
            token = cuda::device::barrier_arrive_tx(*my_barrier, 1, 128*64/2);
        } else {
            token =  my_barrier->arrive();
        }
        if (row < rows) {
            const int s_row = threadIdx.x / 2;

            // condensed version of cvt_quant_to_fp4_get_sf_out_offset that exploits known factors
            int32_t kTileIdx = (col / T) >> 2;
            int32_t numKTiles = cols / 64;
            int32_t mTileIdx = row >> 7;         // mIdx / 128
            auto calc_idx = [&](int k){
                int mIdx = s_row + k * T;
                int32_t outerMIdx = mIdx & 31;        // mIdx % 32
                return (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx) << 9 | (outerMIdx << 4) | ((k/2) << 2);
            };

            #pragma unroll
            for (int k = threadIdx.x % 2; k < T_PER_G; k += 2) {
                int scale_idx = calc_idx(k);
                __pipeline_memcpy_async(scales_fp8 + 4*(s_row + k * T), x_scales + scale_idx, 4);
            }
        }
        __pipeline_commit();
        return token;
    };

    auto wait_token = ldgsts();
    __pipeline_wait_prior(1);
    __syncthreads();

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    m16_n16_b_fragment<nv_bfloat16> b_frags[T_PER_G];
    // load RHT fragments into registers
    for (int k = 0; k < T_PER_G; ++k) {
        b_frags[k] = load_fragment_b_swizzle(lane_id, h_smem + k * T * T);
    }

    while (iter.get<0>() < rows / G) {
        __pipeline_wait_prior(0);
        my_barrier->wait(std::move(wait_token));

        using in_quant_vec = GenericVector<__nv_fp4x2_storage_t, 8>;
        using fp8x4 = GenericVector<__nv_fp8_e4m3, 4>;
        fp8x4 next_group_scales[G/WarpSize];
        in_quant_vec in_vecs[4][G/WarpSize];

        // the compiler cannot prove that the later smem stores don't overlap with these loads,
        // causing bad codegen. Let's manually move all loads to the top here.
        {
            const int s_row = threadIdx.x;
            #pragma unroll
            for (int g = 0; g < 2; ++g) {
                for (int k = 0; k < G / WarpSize; ++k) {
                    int raw = (2 * k * T + s_row) * 2 + g;
                    int y = raw / 8;
                    int x = raw % 8;
                    __nv_fp4x2_storage_t* smem_base = a_q_smem + y * 128 + 16 * (x^(y%2));
                    auto tmp =  GenericVector<__nv_fp4x2_storage_t, 16>::load(smem_base);
                    for (int s = 0; s < 8; ++s) {
                        in_vecs[2*g+0][k][s] = tmp[s+0];
                        in_vecs[2*g+1][k][s] = tmp[s+8];
                    }
                }
            }
            for (int k = 0; k < G/WarpSize; ++k) {
                next_group_scales[k] = fp8x4::load(scales_fp8 + 4*(2 * k * T + s_row));
            }
        }

        __syncwarp();  // everyone has finished loading the data and we can reuse smem
        int row = iter.get<0>() * G;
        int col = iter.get<1>();
        iter.advance();
        wait_token = ldgsts();

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            // dequantize
            #pragma unroll
            for (int k = 0; k < G/WarpSize; ++k) {
                const int s_row = threadIdx.x;
                float group_scale = static_cast<float>(next_group_scales[k][g]) * global_scale;
                float2 scale_v = {group_scale, group_scale};
                for (int c = 0; c < 2; ++c) {
                    auto quants = in_vecs[g][k];
                    GenericVector<nv_bfloat16, 8> res;
                    for (int s = 0; s < 4; ++s) {
                        float2 dq = __nv_cvt_fp4x2_to_float2(quants[s + 4*c]);
                        float2 p = __fmul2_rn(dq, scale_v);
                        res[2*s+0] = p.x;
                        res[2*s+1] = p.y;
                    }
                    res.store(a_smem + 2*k * T * T + (s_row/16) * T * T + swizzle_smem(s_row%16, c));
                }
            }

            __syncwarp();   // everyone has finished loading and storing

            for (int k = 0; k < T_PER_G; ++k) {
                a_frags[k] = load_fragment_aT_swizzle(lane_id, a_smem + k * T * T);
            }

            // now generate all tile-wise products
            m16_n16_k32_c_fragment<float> tile_prods[T_PER_G];
            had128_mma(a_frags, b_frags, tile_prods);
            eden_rounding_tp(y, scales, local_scale_max, tile_prods, row, col+T*g, rows, cols, inv_fp4_max);
        }
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
void rht128_requant_launcher(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N)
{
    if (N % 128 != 0) throw std::runtime_error("rht128_requant: N must be divisible by 128");
    if (M % 128 != 0) throw std::runtime_error("rht128_requant: M must be divisible by 128");

    int blocks, device;
    int smem = NUM_WARPS * 16 * 128 * 2;
    smem += NUM_WARPS * 16 * 128 * 2;
    smem += NUM_WARPS * 128 * 4;
    CUDA_CHECK(cudaGetDevice(&device));

    CUDA_CHECK(cudaFuncSetAttribute(rht128_requant_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rht128_requant_kernel, 32*NUM_WARPS, smem));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaMemset(max_scale, 0, sizeof(unsigned)));

    auto tmap = create_xT_fp4_tensor_map(x, M, N);

    rht128_requant_kernel<<<sms * blocks, dim3(32, NUM_WARPS), smem>>>(
        y, scratch_scales, max_scale, h, tmap, x_scales, x_global_scale, M, N, 1.f / fp4_max);
    CUDA_CHECK(cudaGetLastError());
    launch_eden_convert_scales_kernel(scales_fp8, global_scale_ptr, scratch_scales, max_scale, seed, 1.f / fp8_max, N, M/16);
}

void rht128_requant(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N)
{
    rht128_requant_launcher(y, scales_fp8, global_scale_ptr, scratch_scales, max_scale, h, x, x_scales, x_global_scale, seed, fp4_max, fp8_max, M, N);
}

