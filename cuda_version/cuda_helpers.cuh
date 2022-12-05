#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <device_launch_parameters.h>
//#include <device_functions.h>

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s in %s, line %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}


// For GPU Parallelisation, match this to maximum of computing GPU
#define THREADS 256 // Maximum Threads per Block
#define BLOCKS 92 // Maximum blocks for all SMs in GPU

#endif //CUDA_HELPERS_CUH