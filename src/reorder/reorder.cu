#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "reorder.cuh"

void reorder(at::Tensor X, at::Tensor Y,
    int64_t BM, int64_t BN, int64_t rldn, at::Tensor RA){

    // X: [bs, dim]
    // Y: [bs, dim]

    int bs = X.size(0);
    int dim = X.size(1);

    reorder_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(Y.data_ptr<at::Half>()),
        bs, dim, BM, BN, (dim / BN), rldn, 
        RA.data_ptr<int>()
    );
}