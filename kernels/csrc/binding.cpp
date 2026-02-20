// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>

namespace nb = nanobind;

#define CHECK(X) if(!(X)) throw std::runtime_error(#X " failed");

static std::string to_string(const nb::dlpack::dtype& dt) {
    return "dtype(" + std::to_string(dt.code) + ", " + std::to_string(dt.bits) + ", " + std::to_string(dt.lanes) + ")";
}

template<typename T>
static void check_eq(T a, T b, const char* msg = nullptr) {
    using std::to_string;
    using ::to_string;
    if (a != b) throw std::runtime_error("Assertion `" + std::string(msg) + "` failed: " + to_string(a) + " != " + to_string(b));
}
#define CHECK_EQ(A, B) check_eq(A, B, #A " == " #B)

template<typename... Args>
using CudaArray = nb::ndarray<Args..., nb::c_contig, nb::device::cuda>;

void group_transform_128_eden(__nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
                              nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
                              long seed, float fp4_max, float fp8_max, int M, int N, bool transpose);
extern void group_transform_128(nv_bfloat16* y, const nv_bfloat16* trans, const nv_bfloat16* x, int M, int N, bool transpose);

void transform_rht128(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);
void transform_rht128_ws(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);
void transform_rht128_tma(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);

void transform_rht128_eden(__nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N, bool transpose);

void rht128_requant(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N);


void eden_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long seed, long rows, long cols);
void rtn_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long rows, long cols);
void four_six_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long rows, long cols);
void dequant_tp_had_quant(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N);

void group_transform_128_binding(
    const CudaArray<nb::shape<-1, -1>>& y,
    const CudaArray<nb::ro, nb::shape<128, 128>>& h,
    const CudaArray<nb::ro, nb::shape<-1, -1>>& x,
    bool transpose)
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    if (transpose) {
        CHECK_EQ(x.shape(0), y.shape(1));
        CHECK_EQ(x.shape(1), y.shape(0));
    } else {
        CHECK_EQ(x.shape(0), y.shape(0));
        CHECK_EQ(x.shape(1), y.shape(1));
    }

    CHECK_EQ(x.shape(0) % 128, 0ul);
    CHECK_EQ(x.shape(1) % 128, 0ul);

    CHECK_EQ(y.dtype(), bf16_dt);
    CHECK_EQ(h.dtype(), bf16_dt);
    CHECK_EQ(x.dtype(), bf16_dt);

    group_transform_128(
        reinterpret_cast<nv_bfloat16*>(y.data()),
        reinterpret_cast<const nv_bfloat16*>(h.data()),
        reinterpret_cast<const nv_bfloat16*>(x.data()),
        x.shape(0), x.shape(1), transpose);
}

void transform_rht128_binding(
    const CudaArray<nb::shape<-1, -1>>& y,
    const CudaArray<nb::ro>& h,
    const CudaArray<nb::ro, nb::shape<-1, -1>>& x,
    bool transpose)
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    if (transpose) {
        CHECK_EQ(x.shape(0), y.shape(1));
        CHECK_EQ(x.shape(1), y.shape(0));
    } else {
        CHECK_EQ(x.shape(0), y.shape(0));
        CHECK_EQ(x.shape(1), y.shape(1));
    }

    CHECK_EQ(x.shape(0) % 128, 0ul);
    CHECK_EQ(x.shape(1) % 128, 0ul);

    CHECK_EQ(y.dtype(), bf16_dt);
    CHECK_EQ(h.dtype(), bf16_dt);
    CHECK_EQ(x.dtype(), bf16_dt);
    CHECK_EQ(h.size(), 16*16*8ul);

    transform_rht128_tma(
        reinterpret_cast<nv_bfloat16*>(y.data()),
        reinterpret_cast<const nv_bfloat16*>(h.data()),
        reinterpret_cast<const nv_bfloat16*>(x.data()),
        x.shape(0), x.shape(1), transpose);
}

void group_transform_128_eden_binding(
    const CudaArray<unsigned char>& y,
    const CudaArray<>& group_scales,
    const CudaArray<float, nb::shape<>>& tensor_scale,
    const CudaArray<nb::shape<-1>>& scratch,
    const CudaArray<unsigned, nb::shape<>>& max_scale,
    const CudaArray<nb::ro, nb::shape<128, 128>>& h,
    const CudaArray<nb::ro>& x,
    const nb::ndarray<long, nb::c_contig, nb::device::cpu, nb::shape<>>& seed,
    float fp4_max, float fp8_max, bool transpose)
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(x.ndim(), 2ul);
    CHECK_EQ(x.shape(0) % 128, 0ul);
    CHECK_EQ(x.shape(1) % 128, 0ul);
    CHECK_EQ(h.dtype(), bf16_dt);
    CHECK_EQ(x.dtype(), bf16_dt);

    if (transpose) {
        CHECK_EQ(y.shape(1), x.shape(0) / 2);
        CHECK_EQ(y.shape(0), x.shape(1));
        CHECK_EQ(group_scales.shape(1), x.shape(0) / 16);
        CHECK_EQ(group_scales.shape(0), x.shape(1));
    } else {
        CHECK_EQ(y.shape(0), x.shape(0));
        CHECK_EQ(y.shape(1), x.shape(1) / 2);
        CHECK_EQ(group_scales.shape(0), x.shape(0));
        CHECK_EQ(group_scales.shape(1), x.shape(1) / 16);
    }

    CHECK_EQ(scratch.shape(0), x.size() / 16);
    CHECK_EQ(scratch.dtype(), bf16_dt);

    group_transform_128_eden(
        y.data(),
        reinterpret_cast<__nv_fp8_e4m3*>(group_scales.data()),
        tensor_scale.data(),
        reinterpret_cast<nv_bfloat16*>(scratch.data()),
        max_scale.data(),
        reinterpret_cast<const nv_bfloat16*>(h.data()),
        reinterpret_cast<const nv_bfloat16*>(x.data()),
        *seed.data(),  fp4_max, fp8_max,
        x.shape(0), x.shape(1), transpose);
}

void rht_128_eden_binding(
    const CudaArray<unsigned char>& y,
    const CudaArray<>& group_scales,
    const CudaArray<float, nb::shape<>>& tensor_scale,
    const CudaArray<nb::shape<-1>>& scratch,
    const CudaArray<unsigned, nb::shape<>>& max_scale,
    const CudaArray<nb::ro>& h,
    const CudaArray<nb::ro>& x,
    const nb::ndarray<long, nb::c_contig, nb::device::cpu, nb::shape<>>& seed,
    float fp4_max, float fp8_max, bool transpose)
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(x.ndim(), 2ul);
    CHECK_EQ(x.shape(0) % 128, 0ul);
    CHECK_EQ(x.shape(1) % 128, 0ul);
    CHECK_EQ(h.dtype(), bf16_dt);
    CHECK_EQ(x.dtype(), bf16_dt);
    CHECK_EQ(h.size(), 16*16*8ul);

    if (transpose) {
        CHECK_EQ(y.shape(1), x.shape(0) / 2);
        CHECK_EQ(y.shape(0), x.shape(1));
        CHECK_EQ(group_scales.shape(1), x.shape(0) / 16);
        CHECK_EQ(group_scales.shape(0), x.shape(1));
    } else {
        CHECK_EQ(y.shape(0), x.shape(0));
        CHECK_EQ(y.shape(1), x.shape(1) / 2);
        CHECK_EQ(group_scales.shape(0), x.shape(0));
        CHECK_EQ(group_scales.shape(1), x.shape(1) / 16);
    }

    CHECK_EQ(scratch.shape(0), x.size() / 16);
    CHECK_EQ(scratch.dtype(), bf16_dt);

    transform_rht128_eden(
        y.data(),
        reinterpret_cast<__nv_fp8_e4m3*>(group_scales.data()),
        tensor_scale.data(),
        reinterpret_cast<nv_bfloat16*>(scratch.data()),
        max_scale.data(),
        reinterpret_cast<const nv_bfloat16*>(h.data()),
        reinterpret_cast<const nv_bfloat16*>(x.data()),
        *seed.data(),  fp4_max, fp8_max,
        x.shape(0), x.shape(1), transpose);
}

void eden_fp4_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nb::shape<>>& global_scale,
    const CudaArray<nb::ro>& inp,
    const CudaArray<float, nb::ro>& amax_ptr,
    float scale_override,
    const nb::ndarray<long, nb::c_contig, nb::device::cpu, nb::shape<>>& seed
    )
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(inp.ndim(), 2ul);
    CHECK_EQ(out.ndim(), 2ul);

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(out.size(), inp.size() / 2);
    CHECK_EQ(out.dtype().bits, static_cast<uint8_t>(8));
    CHECK_EQ(scales.size(), inp.size() / 16);
    CHECK_EQ(scales.dtype().bits, static_cast<uint8_t>(8));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK(global_scale.data() != amax_ptr.data());

    eden_fp4(
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        amax_ptr.data(),
        scale_override, *seed.data(), inp.shape(0), inp.shape(1));
}

void four_six_fp4_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nb::shape<>>& global_scale,
    const CudaArray<nb::ro>& inp,
    const CudaArray<float, nb::ro>& amax_ptr,
    float scale_override
    )
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(inp.ndim(), 2ul);
    CHECK_EQ(out.ndim(), 2ul);

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(out.size(), inp.size() / 2);
    CHECK_EQ(out.dtype().bits, static_cast<uint8_t>(8));
    CHECK_EQ(scales.size(), inp.size() / 16);
    CHECK_EQ(scales.dtype().bits, static_cast<uint8_t>(8));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK(global_scale.data() != amax_ptr.data());

    four_six_fp4(
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        amax_ptr.data(),
        scale_override, inp.shape(0), inp.shape(1));
}

void rtn_fp4_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nb::shape<>>& global_scale,
    const CudaArray<nb::ro>& inp,
    const CudaArray<float, nb::ro>& amax_ptr,
    float scale_override
    )
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(inp.ndim(), 2ul);
    CHECK_EQ(out.ndim(), 2ul);

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(out.size(), inp.size() / 2);
    CHECK_EQ(out.dtype().bits, static_cast<uint8_t>(8));
    CHECK_EQ(scales.size(), inp.size() / 16);
    CHECK_EQ(scales.dtype().bits, static_cast<uint8_t>(8));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK(global_scale.data() != amax_ptr.data());

    rtn_fp4(
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        amax_ptr.data(),
        scale_override, inp.shape(0), inp.shape(1));
}


void dequant_tp_had_quant_binding(
    const CudaArray<unsigned char>& y,
    const CudaArray<>& group_scales,
    const CudaArray<float, nb::shape<>>& tensor_scale,
    const CudaArray<nb::shape<-1>>& scratch,
    const CudaArray<unsigned, nb::shape<>>& max_scale,
    const CudaArray<nb::ro, nb::shape<128, 128>>& h,
    const CudaArray<unsigned char, nb::ro>& x,
    const CudaArray<unsigned char, nb::ro>& x_micro_scales,
    const CudaArray<float, nb::ro, nb::shape<>>& x_global_scale,
    const nb::ndarray<long, nb::c_contig, nb::device::cpu, nb::shape<>>& seed,
    float fp4_max, float fp8_max)
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(x.ndim(), 2ul);
    CHECK_EQ(x.shape(0) % 128, 0ul);
    CHECK_EQ(x.shape(1) % 64, 0ul);
    CHECK_EQ(h.dtype(), bf16_dt);

    CHECK_EQ(y.shape(1), x.shape(0) / 2);
    CHECK_EQ(y.shape(0), x.shape(1) * 2);
    CHECK_EQ(group_scales.shape(1), x.shape(0) / 16);
    CHECK_EQ(group_scales.shape(0), x.shape(1) * 2);

    CHECK_EQ(x_micro_scales.shape(0), x.shape(0));
    CHECK_EQ(x_micro_scales.shape(1), x.shape(1) / 8);

    CHECK_EQ(scratch.shape(0), x.size() / 8);
    CHECK_EQ(scratch.dtype(), bf16_dt);

    dequant_tp_had_quant(
        y.data(),
        reinterpret_cast<__nv_fp8_e4m3*>(group_scales.data()),
        tensor_scale.data(),
        reinterpret_cast<nv_bfloat16*>(scratch.data()),
        max_scale.data(),
        reinterpret_cast<const nv_bfloat16*>(h.data()),
        x.data(),
        reinterpret_cast<const __nv_fp8_e4m3*>(x_micro_scales.data()),
        x_global_scale.data(),
        *seed.data(),  fp4_max, fp8_max,
        x.shape(0), x.shape(1) * 2);
}



void rht128_requant_binding(
    const CudaArray<unsigned char>& y,
    const CudaArray<>& group_scales,
    const CudaArray<float, nb::shape<>>& tensor_scale,
    const CudaArray<nb::shape<-1>>& scratch,
    const CudaArray<unsigned, nb::shape<>>& max_scale,
    const CudaArray<nb::ro>& h,
    const CudaArray<unsigned char, nb::ro>& x,
    const CudaArray<unsigned char, nb::ro>& x_micro_scales,
    const CudaArray<float, nb::ro, nb::shape<>>& x_global_scale,
    const nb::ndarray<long, nb::c_contig, nb::device::cpu, nb::shape<>>& seed,
    float fp4_max, float fp8_max)
{
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(x.ndim(), 2ul);
    CHECK_EQ(x.shape(0) % 128, 0ul);
    CHECK_EQ(x.shape(1) % 64, 0ul);
    CHECK_EQ(h.dtype(), bf16_dt);

    CHECK_EQ(y.shape(1), x.shape(0) / 2);
    CHECK_EQ(y.shape(0), x.shape(1) * 2);
    CHECK_EQ(group_scales.shape(1), x.shape(0) / 16);
    CHECK_EQ(group_scales.shape(0), x.shape(1) * 2);

    CHECK_EQ(x_micro_scales.shape(0), x.shape(0));
    CHECK_EQ(x_micro_scales.shape(1), x.shape(1) / 8);

    CHECK_EQ(scratch.shape(0), x.size() / 8);
    CHECK_EQ(scratch.dtype(), bf16_dt);
    CHECK_EQ(h.size(), 16*16*8ul);

    rht128_requant(
        y.data(),
        reinterpret_cast<__nv_fp8_e4m3*>(group_scales.data()),
        tensor_scale.data(),
        reinterpret_cast<nv_bfloat16*>(scratch.data()),
        max_scale.data(),
        reinterpret_cast<const nv_bfloat16*>(h.data()),
        x.data(),
        reinterpret_cast<const __nv_fp8_e4m3*>(x_micro_scales.data()),
        x_global_scale.data(),
        *seed.data(),  fp4_max, fp8_max,
        x.shape(0), x.shape(1) * 2);
}


NB_MODULE(_quartet2, m) {
    m.def("group_transform_128", &group_transform_128_binding);
    m.def("transform_rht128", &transform_rht128_binding);
    m.def("group_transform_128_eden", &group_transform_128_eden_binding, nb::arg("out"), nb::arg("group_scales"),
        nb::arg("tensor_scale"), nb::arg("scratch"), nb::arg("max_scale"), nb::arg("h"), nb::arg("x"), nb::arg("seed"),
        nb::arg("fp4_max") = 6, nb::arg("fp8_max") = 256, nb::arg("transpose") = false);
    m.def("rht128_eden", &rht_128_eden_binding, nb::arg("out"), nb::arg("group_scales"),
        nb::arg("tensor_scale"), nb::arg("scratch"), nb::arg("max_scale"), nb::arg("h"), nb::arg("x"), nb::arg("seed"),
        nb::arg("fp4_max") = 6, nb::arg("fp8_max") = 256, nb::arg("transpose") = false);

    m.def("dequant_tp_had_quant", &dequant_tp_had_quant_binding,
        nb::arg("out"), nb::arg("out_group_scales"), nb::arg("out_tensor_scale"),
        nb::arg("scratch"), nb::arg("max_scale"), nb::arg("h"),
        nb::arg("x"), nb::arg("x_group_scales"), nb::arg("x_tensor_scale"),
        nb::arg("seed"), nb::arg("fp4_max") = 6, nb::arg("fp8_max") = 256);

    m.def("rht128_requant", &rht128_requant_binding,
        nb::arg("out"), nb::arg("out_group_scales"), nb::arg("out_tensor_scale"),
        nb::arg("scratch"), nb::arg("max_scale"), nb::arg("h"),
        nb::arg("x"), nb::arg("x_group_scales"), nb::arg("x_tensor_scale"),
        nb::arg("seed"), nb::arg("fp4_max") = 6, nb::arg("fp8_max") = 256);

    m.def("eden_fp4", &eden_fp4_binding,  nb::arg("out"), nb::arg("scales"), nb::arg("global_scale"), nb::arg("input"), nb::arg("amax"), nb::arg("scale_override"), nb::arg("seed"));
    m.def("four_six_fp4", &four_six_fp4_binding, nb::arg("out"), nb::arg("scales"), nb::arg("global_scale"), nb::arg("input"), nb::arg("amax"), nb::arg("scale_override"));
    m.def("rtn_fp4", &rtn_fp4_binding, nb::arg("out"), nb::arg("scales"), nb::arg("global_scale"), nb::arg("input"), nb::arg("amax"), nb::arg("scale_override"));
}
