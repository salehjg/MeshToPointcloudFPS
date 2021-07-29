#pragma once

#include <cuda_runtime.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

extern
void farthestpointsamplingLauncher(
    int b,
    int n,
    int m,
    const float * inp,
    float * temp,
    int * out);