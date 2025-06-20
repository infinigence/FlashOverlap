#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "../rmsnorm/utils.h"
/*
    Reorder kernel for correctness check. 
*/
__global__ __forceinline__ void reorder_kernel(
                    half* x, half* o, 
                    int bs, int dim, int64_t BM, int64_t BN, 
                    int64_t ldn, int64_t rldn, int* RA){

  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  // perform a block-wise reorder here
  int old_index = bid / BM * ldn + j / BN;
  int new_index = RA[old_index];
  int new_row = new_index / rldn * BM + bid % BM;
  int new_col = new_index % rldn * BN + j % BN;

  *(float4*)(&o[bid * dim + j]) 
    = *(float4*)(&x[new_row * (rldn * BN) + new_col]);
}