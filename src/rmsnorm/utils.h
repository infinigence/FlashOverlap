/*
    Utility functions. 
*/

#pragma once

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__device__ __forceinline__ float warpReduceSum(float sum_val,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 1);  // 0-1, 2-3, 4-5, etc.
  return sum_val;
}

__device__ __forceinline__ half warpReduceSum(half result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ half2 warpReduceSum(half2 result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ float warpReduceMax(float max_val,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 1));  // 0-1, 2-3, 4-5, etc.
  return max_val;
}

__device__ __forceinline__ float blockReduceSum(float reducing, float *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    const int32_t WPT = blockDim.x / 32;
    int32_t WPTB = 32 / (32 / WPT);
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

# pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPTB) reducing = lane_id < WPT ? shared_mem[lane_id] : 0.0f;

# pragma unroll
    for (int32_t mask = WPTB / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

__device__ __forceinline__ half blockReduceSum(half reducing, half *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    const int32_t WPT = blockDim.x / 32;
    int32_t WPTB = 32 / (32 / WPT);
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

# pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPTB) reducing = lane_id < WPT ? shared_mem[lane_id] : 0.0f;

# pragma unroll
    for (int32_t mask = WPTB / 2; mask >= 1; mask /= 2) {
        reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}