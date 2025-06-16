#pragma once

#include <nccl.h>
#include <vector>
#include <cublas_v2.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

class BaselineImpl : public torch::CustomClassHolder {
    public:
        BaselineImpl();
        ~BaselineImpl();

        void NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id);
        void CublasInit();

        void GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C);
        void GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D);
        void GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor mLen_CPU);
        void Gemm(at::Tensor A, at::Tensor B, at::Tensor C);

        void NcclAllReduce(at::Tensor C);
        void NcclReduceScatter(at::Tensor C);
        void NcclAll2All(at::Tensor C, at::Tensor D, at::Tensor mLen_CPU);
        
    private:
        ncclComm_t comm;
        int64_t my_rank;
        int64_t my_size;

        cublasHandle_t my_handle;
        cudaStream_t my_stream;
};