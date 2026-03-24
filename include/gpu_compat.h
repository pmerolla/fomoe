/*
 * gpu_compat.h — Thin compatibility layer for HIP <-> CUDA
 *
 * All GPU code is written using HIP API names (hip*).
 * When building for NVIDIA (FR_CUDA), this header maps them to CUDA equivalents.
 * When building for AMD (default), it just includes the native HIP headers.
 */

#ifndef FR_GPU_COMPAT_H
#define FR_GPU_COMPAT_H

#ifdef FR_CUDA
// ============================================================
//  NVIDIA CUDA backend
// ============================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// --- Types ---
#define hipStream_t             cudaStream_t
#define hipEvent_t              cudaEvent_t
#define hipError_t              cudaError_t
#define hipDeviceProp_t         cudaDeviceProp

// --- Constants ---
#define hipSuccess              cudaSuccess
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipHostMallocDefault    cudaHostAllocDefault
#define hipHostRegisterDefault  cudaHostRegisterDefault

// --- Device management ---
#define hipGetDeviceCount       cudaGetDeviceCount
#define hipSetDevice            cudaSetDevice
#define hipGetDeviceProperties  cudaGetDeviceProperties
#define hipGetErrorString       cudaGetErrorString
#define hipGetLastError         cudaGetLastError
#define hipDeviceSynchronize    cudaDeviceSynchronize

// --- Stream management ---
#define hipStreamCreate         cudaStreamCreate
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamDestroy        cudaStreamDestroy
#define hipStreamSynchronize    cudaStreamSynchronize
#define hipStreamWaitEvent      cudaStreamWaitEvent
#define hipStreamNonBlocking    cudaStreamNonBlocking

// --- Event management ---
#define hipEventCreate          cudaEventCreate
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDestroy         cudaEventDestroy
#define hipEventRecord          cudaEventRecord
#define hipEventSynchronize     cudaEventSynchronize
#define hipEventElapsedTime     cudaEventElapsedTime
#define hipEventDisableTiming   cudaEventDisableTiming

// --- Device memory ---
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemcpy               cudaMemcpy
#define hipMemcpyAsync          cudaMemcpyAsync
#define hipMemset               cudaMemset
#define hipMemsetAsync          cudaMemsetAsync
#define hipMemGetInfo           cudaMemGetInfo

// --- Host/pinned memory ---
template<typename T>
static inline cudaError_t _fr_hipHostMalloc(T **ptr, size_t size,
                                             unsigned int flags = 0) {
    return cudaHostAlloc((void **)ptr, size, flags);
}
#define hipHostMalloc           _fr_hipHostMalloc
#define hipHostFree             cudaFreeHost
#define hipHostRegister         cudaHostRegister
#define hipHostUnregister       cudaHostUnregister

// --- Warp shuffles ---
#define __shfl_down(val, offset)    __shfl_down_sync(0xffffffff, (val), (offset))
#define __shfl_xor(val, offset)     __shfl_xor_sync(0xffffffff, (val), (offset))
#define __shfl(val, src)            __shfl_sync(0xffffffff, (val), (src))

// --- Kernel attributes ---
#define hipFuncSetAttribute         cudaFuncSetAttribute
#define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize

#else
// ============================================================
//  AMD ROCm / HIP backend (native)
// ============================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#endif // FR_CUDA

// ============================================================
//  Common helpers
// ============================================================

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define HIP_CHECK_I(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#endif // FR_GPU_COMPAT_H
