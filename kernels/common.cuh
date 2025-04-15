
#include "cuda_fp8.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"

#define F8E4M3_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3))

// TODO: This is often used to check that the data is contiguous so that
// kernels can be easily mapped. However this only returns true for row
// major, if all the inputs are column major, we could apply the fast path
// too (but we wouldn't if some of them are row major and some column major).
__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}


__device__ __forceinline__ bool isnang(float a) { return isnan(a); }
__device__ __forceinline__ bool isnang(double a) { return isnan(a); }
__device__ __forceinline__ float recipg(float a) { return 1.0 / a; }
__device__ __forceinline__ double recipg(double a) { return 1.0 / a; }
__device__ __forceinline__ float cosg(float a) { return cosf(a); }
__device__ __forceinline__ double cosg(double a) { return cos(a); }
__device__ __forceinline__ float sing(float a) { return sinf(a); }
__device__ __forceinline__ double sing(double a) { return sin(a); }
__device__ __forceinline__ float sqrtg(float a) { return sqrtf(a); }
__device__ __forceinline__ double sqrtg(double a) { return sqrt(a); }
__device__ __forceinline__ float powg(float a, float b) { return powf(a, b); }
__device__ __forceinline__ double powg(double a, double b) { return pow(a, b); }
__device__ __forceinline__ float tanhg(float a) { return tanhf(a); }
__device__ __forceinline__ double tanhg(double a) { return tanh(a); }
__device__ __forceinline__ float erfg(float a) { return erff(a); }
__device__ __forceinline__ double erfg(double a) { return erf(a); }
__device__ __forceinline__ float ceilg(float a) { return ceilf(a); }
__device__ __forceinline__ double ceilg(double a) { return ceil(a); }
__device__ __forceinline__ float floorg(float a) { return floorf(a); }
__device__ __forceinline__ double floorg(double a) { return floor(a); }
__device__ __forceinline__ float roundg(float a) { return roundf(a); }
__device__ __forceinline__ double roundg(double a) { return round(a); }
__device__ __forceinline__ float normcdfg(float a) { return normcdff(a); }
__device__ __forceinline__ double normcdfg(double a) { return normcdf(a); }
__device__ __forceinline__ float maxg(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double maxg(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ float ming(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ double ming(double a, double b) { return fmin(a, b); }
__device__ __forceinline__ float logg(float a) { return logf(a); }
__device__ __forceinline__ double logg(double a) { return log(a); }
__device__ __forceinline__ float expg(float a) { return expf(a); }
__device__ __forceinline__ double expg(double a) { return exp(a); }
__device__ __forceinline__ float absg(float a) { return fabsf(a); }
__device__ __forceinline__ double absg(double a) { return fabs(a); }
__device__ __forceinline__ float copysigng(float a, float b) { return copysignf(a, b); }
__device__ __forceinline__ double copysigng(double a, double b) { return copysign(a, b); }

__device__ __forceinline__ int16_t ming(int16_t a, int16_t b) { return min(a, b); }
__device__ __forceinline__ int16_t maxg(int16_t a, int16_t b) { return max(a, b); }
__device__ __forceinline__ int32_t ming(int32_t a, int32_t b) { return min(a, b); }
__device__ __forceinline__ int32_t maxg(int32_t a, int32_t b) { return max(a, b); }
__device__ __forceinline__ int64_t ming(int64_t a, int64_t b) { return min(a, b); }
__device__ __forceinline__ int64_t maxg(int64_t a, int64_t b) { return max(a, b); }
__device__ __forceinline__ uint32_t ming(uint32_t a, uint32_t b) { return min(a, b); }
__device__ __forceinline__ uint32_t maxg(uint32_t a, uint32_t b) { return max(a, b); }
__device__ __forceinline__ uint8_t ming(uint8_t a, uint8_t b) { return min(a, b); }
__device__ __forceinline__ uint8_t maxg(uint8_t a, uint8_t b) { return max(a, b); }

__device__ __forceinline__ __half powg(__half a, __half b) { return __float2half(powf(__half2float(a), __half2float(b))); }
__device__ __forceinline__ bool isnang(__half a) { return __hisnan(a); }
__device__ __forceinline__ __half sqrtg(__half a) { return hsqrt(a); }
__device__ __forceinline__ __half cosg(__half a) { return hcos(a); }
__device__ __forceinline__ __half sing(__half a) { return hsin(a); }
__device__ __forceinline__ __half recipg(__half a) { __half one = 1.0; return one / a; }
__device__ __forceinline__ __half maxg(__half a, __half b) { return __hmax_nan(a, b); }
__device__ __forceinline__ __half tanhg(__half a) { return __float2half(tanhf(__half2float(a))); }
__device__ __forceinline__ __half erfg(__half a) { return __float2half(erff(__half2float(a))); }
__device__ __forceinline__ __half ceilg(__half a) { return __float2half(ceilf(__half2float(a))); }
__device__ __forceinline__ __half floorg(__half a) { return __float2half(floorf(__half2float(a))); }
__device__ __forceinline__ __half roundg(__half a) { return __float2half(roundf(__half2float(a))); }
__device__ __forceinline__ __half normcdfg(__half a) { return __float2half(normcdff(__half2float(a))); }
__device__ __forceinline__ __half ming(__half a, __half b) { return __hmin_nan(a, b); }
__device__ __forceinline__ __half logg(__half a) { return hlog(a); }
__device__ __forceinline__ __half expg(__half a) { return hexp(a); }
__device__ __forceinline__ __half absg(__half a) { return __habs(a); }
__device__ __forceinline__ __half copysigng(__half a, __half b) { return __float2half(copysignf(__half2float(a), __half2float(b))); }


__device__ __forceinline__ __nv_bfloat16 powg(__nv_bfloat16 a, __nv_bfloat16 b) { return __float2bfloat16(powf(__bfloat162float(a), __bfloat162float(b))); }
__device__ __forceinline__ bool isnang(__nv_bfloat16 a) { return __hisnan(a); }
__device__ __forceinline__ __nv_bfloat16 sqrtg(__nv_bfloat16 a) { return hsqrt(a); }
__device__ __forceinline__ __nv_bfloat16 cosg(__nv_bfloat16 a) { return hcos(a); }
__device__ __forceinline__ __nv_bfloat16 sing(__nv_bfloat16 a) { return hsin(a); }
__device__ __forceinline__ __nv_bfloat16 recipg(__nv_bfloat16 a) { __nv_bfloat16 one = 1.0; return one / a; }
__device__ __forceinline__ __nv_bfloat16 maxg(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax_nan(a, b); }
__device__ __forceinline__ __nv_bfloat16 tanhg(__nv_bfloat16 a) { return __float2bfloat16(tanhf(__bfloat162float(a))); }
__device__ __forceinline__ __nv_bfloat16 erfg(__nv_bfloat16 a) { return __float2bfloat16(erff(__bfloat162float(a))); }
__device__ __forceinline__ __nv_bfloat16 ceilg(__nv_bfloat16 a) { return __float2bfloat16(ceilf(__bfloat162float(a))); }
__device__ __forceinline__ __nv_bfloat16 floorg(__nv_bfloat16 a) { return __float2bfloat16(floorf(__bfloat162float(a))); }
__device__ __forceinline__ __nv_bfloat16 roundg(__nv_bfloat16 a) { return __float2bfloat16(roundf(__bfloat162float(a))); }
__device__ __forceinline__ __nv_bfloat16 normcdfg(__nv_bfloat16 a) { return __float2bfloat16(normcdff(__bfloat162float(a))); }
__device__ __forceinline__ __nv_bfloat16 ming(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin_nan(a, b); }
__device__ __forceinline__ __nv_bfloat16 logg(__nv_bfloat16 a) { return hlog(a); }
__device__ __forceinline__ __nv_bfloat16 expg(__nv_bfloat16 a) { return hexp(a); }
__device__ __forceinline__ __nv_bfloat16 absg(__nv_bfloat16 a) { return __habs(a); }
__device__ __forceinline__ __nv_bfloat16 copysigng(__nv_bfloat16 a, __nv_bfloat16 b) { return __float2bfloat16(copysignf(__bfloat162float(a), __bfloat162float(b))); }

__device__ __forceinline__ __nv_fp8_e4m3 powg(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { return __nv_fp8_e4m3(powf(F8E4M3_TO_FLOAT(a), F8E4M3_TO_FLOAT(b))); }
__device__ __forceinline__ bool isnang(__nv_fp8_e4m3 a) { return isnan(F8E4M3_TO_FLOAT(a)); }
__device__ __forceinline__ __nv_fp8_e4m3 sqrtg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(sqrtf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 cosg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(cosf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 sing(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(sinf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 recipg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(1. / F8E4M3_TO_FLOAT(a)); }
__device__ __forceinline__ __nv_fp8_e4m3 maxg(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { return __nv_fp8_e4m3(fmaxf(F8E4M3_TO_FLOAT(a), F8E4M3_TO_FLOAT(b))); }
__device__ __forceinline__ __nv_fp8_e4m3 tanhg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(tanhf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 erfg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(erff(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 ceilg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(ceilf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 floorg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(floorf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 roundg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(roundf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 normcdfg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(normcdff(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 ming(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { return __nv_fp8_e4m3(fminf(F8E4M3_TO_FLOAT(a), F8E4M3_TO_FLOAT(b))); }
__device__ __forceinline__ __nv_fp8_e4m3 logg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(logf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 expg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(expf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 absg(__nv_fp8_e4m3 a) { return __nv_fp8_e4m3(fabsf(F8E4M3_TO_FLOAT(a))); }
__device__ __forceinline__ __nv_fp8_e4m3 copysigng(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { return __nv_fp8_e4m3(copysignf(F8E4M3_TO_FLOAT(a), F8E4M3_TO_FLOAT(b))); }

