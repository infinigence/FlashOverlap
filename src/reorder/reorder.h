#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void reorder(at::Tensor X, at::Tensor Y,
    int64_t BM, int64_t BN, int64_t rldn, at::Tensor RA);