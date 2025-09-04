#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "rmsnorm.cuh"

void reorder_rmsnorm(at::Tensor X, at::Tensor RX, at::Tensor RW, 
    int64_t BM, int64_t BN, int64_t rldn, at::Tensor RA){

  // X: [bs, dim]
  // RX: [bs, dim]
  
  int bs = X.size(0);
  int dim = X.size(1);

  reorder_rmsnorm_kernel<<<dim3(bs), dim3(128, 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RX.data_ptr<at::Half>()),
        bs, dim, BM, BN, (dim / BN), rldn, 
        RA.data_ptr<int>()
    );
}

void rmsnorm(at::Tensor X, at::Tensor O, at::Tensor RW){
  
  // X: [bs, dim]
  // RX: [bs, dim]
  
  int bs = X.size(0);
  int dim = X.size(1);

  rmsnorm_kernel<<<dim3(bs), dim3(128, 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(O.data_ptr<at::Half>()),
        bs, dim
    );
}