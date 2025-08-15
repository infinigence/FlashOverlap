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
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 4;

  if (j >= dim) {return;}

  half2 x_val[8];
  half2 w_val[8];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&x_val[4]) = *(float4*)(&x[bid * dim + j + 8]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);
  *(float4*)(&w_val[4]) = *(float4*)(&rw[j + 8]);

  // RMSNorm (float)
#pragma unroll
  for (int i = 0; i < 8; i++){
    pow_sum += __half2float(x_val[i].x) * __half2float(x_val[i].x);
    pow_sum += __half2float(x_val[i].y) * __half2float(x_val[i].y);
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
  for (int i = 0; i < 8; i++){
    x_val[i].x = __float2half(__half2float(x_val[i].x) * scaling);
    x_val[i].y = __float2half(__half2float(x_val[i].y) * scaling);
    x_val[i] = __hmul2(x_val[i], w_val[i]);
  }
  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
  *(float4*)(&o[bid * dim + j + 8]) = *(float4*)(&x_val[4]);
}

/*
    Reorder + RMSNorm kernel.
*/
__global__ __forceinline__ void reorder_rmsnorm_kernel(
                    half* x, half* rw, half* o, 
                    int bs, int dim, int64_t BM, int64_t BN, 
                    int64_t ldn, int64_t rldn, int* RA){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 4;

  if (j >= dim) {return;}

  half2 x_val[8];
  half2 w_val[8];
  float pow_sum = 0.0f;

  // perform a block-wise reorder here
  int old_index = bid / BM * ldn + j / BN;
  int new_index = RA[old_index];
  int new_row = new_index / rldn * BM + bid % BM;
  int new_col = new_index % rldn * BN + j % BN;

  *(float4*)(&x_val[0]) = *(float4*)(&x[new_row * (rldn * BN) + new_col]);
  *(float4*)(&x_val[4]) = *(float4*)(&x[new_row * (rldn * BN) + new_col + 8]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);
  *(float4*)(&w_val[4]) = *(float4*)(&rw[j + 8]);

  // RMSNorm (float)
#pragma unroll
  for (int i = 0; i < 8; i++){
    pow_sum += __half2float(x_val[i].x) * __half2float(x_val[i].x);
    pow_sum += __half2float(x_val[i].y) * __half2float(x_val[i].y);
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
  for (int i = 0; i < 8; i++){
    x_val[i].x = __float2half(__half2float(x_val[i].x) * scaling);
    x_val[i].y = __float2half(__half2float(x_val[i].y) * scaling);
    x_val[i] = __hmul2(x_val[i], w_val[i]);
  }
  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
  *(float4*)(&o[bid * dim + j + 8]) = *(float4*)(&x_val[4]);
}