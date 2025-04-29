#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void rmsnorm(at::Tensor X, at::Tensor RX, at::Tensor RW);
void reorder_rmsnorm(at::Tensor X, at::Tensor RX, at::Tensor RW, 
    int64_t BM, int64_t BN, int64_t rldn, at::Tensor RA);