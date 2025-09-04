#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.h"

/*
    RMSNorm kernel.
*/
__global__ __forceinline__ void rmsnorm_kernel(
                    half* x, half* rw, half* o, int bs, int dim){
  
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  // int offset = tid << 3;
  int elem_per_thd = dim / blockDim.x;
  int offset = tid * elem_per_thd;

  half2 x_val[64];
  half2 w_val[4];
  float pow_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < elem_per_thd; i += 8){
    int j = offset + i;
    if (j >= dim) {return;}

    *(float4*)(&x_val[(i >> 1)]) = *(float4*)(&x[bid * dim + j]);

    // RMSNorm (float)
  #pragma unroll
    for (int k = 0; k < 4; k++){
      pow_sum += __half2float(x_val[(i >> 1) + k].x) * __half2float(x_val[(i >> 1) + k].x);
      pow_sum += __half2float(x_val[(i >> 1) + k].y) * __half2float(x_val[(i >> 1) + k].y);
    }
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];

#pragma unroll
  for (int i = 0; i < elem_per_thd; i += 8){
    int j = offset + i;
    if (j >= dim) {return;}
    *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  #pragma unroll
    for (int k = 0; k < 4; k++){
      x_val[(i >> 1) + k].x = __float2half(__half2float(x_val[(i >> 1) + k].x) * scaling);
      x_val[(i >> 1) + k].y = __float2half(__half2float(x_val[(i >> 1) + k].y) * scaling);
      x_val[(i >> 1) + k] = __hmul2(x_val[(i >> 1) + k], w_val[k]);
    }

    // store intermediate value
    *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[i >> 1]);
  }
}

/*
    Reorder + RMSNorm kernel.
*/
__global__ __forceinline__ void reorder_rmsnorm_kernel(
                    half* x, half* rw, half* o, 
                    int bs, int dim, int64_t BM, int64_t BN, 
                    int64_t ldn, int64_t rldn, int* RA){
  
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  // int offset = tid << 3;
  int elem_per_thd = dim / blockDim.x;
  int offset = tid * elem_per_thd;

  int old_index = bid / BM * ldn + offset / BN;
  int new_index = RA[old_index];
  int new_row = new_index / rldn * BM + bid % BM;

  half2 x_val[64];
  half2 w_val[4];
  float pow_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < elem_per_thd; i += 8){
    int new_col = new_index % rldn * BN + (i + offset) % BN;
  
    *(float4*)(&x_val[(i >> 1)]) = *(float4*)(&x[new_row * (rldn * BN) + new_col]);

    // RMSNorm (float)
  #pragma unroll
    for (int k = 0; k < 4; k++){
      pow_sum += __half2float(x_val[(i >> 1) + k].x) * __half2float(x_val[(i >> 1) + k].x);
      pow_sum += __half2float(x_val[(i >> 1) + k].y) * __half2float(x_val[(i >> 1) + k].y);
    }
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];

#pragma unroll
  for (int i = 0; i < elem_per_thd; i += 8){
    int j = offset + i;
    if (j >= dim) {return;}
    *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  #pragma unroll
    for (int k = 0; k < 4; k++){
      x_val[(i >> 1) + k].x = __float2half(__half2float(x_val[(i >> 1) + k].x) * scaling);
      x_val[(i >> 1) + k].y = __float2half(__half2float(x_val[(i >> 1) + k].y) * scaling);
      x_val[(i >> 1) + k] = __hmul2(x_val[(i >> 1) + k], w_val[k]);
    }

    // store intermediate value
    *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[i >> 1]);
  }
}