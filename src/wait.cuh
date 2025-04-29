#pragma once

#include <cuda_runtime_api.h>

__global__ __forceinline__ void kernel_wait_flag (const int that, int* addr) {
    while (atomicCAS (addr, that, 0) != that) {
        __nanosleep (100);
    }
}