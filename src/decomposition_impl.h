#pragma once

#include <nccl.h>
#include <vector>
#include <cublas_v2.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

class DecompositionImpl : public torch::CustomClassHolder {
    public:
        DecompositionImpl();
        ~DecompositionImpl();

        void NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id);
        void CublasInit();

        void GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C);
        void GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D);
        void GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor mLen_CPU);

    private:
        cudaStream_t gemm_stream;
        cudaStream_t comm_stream;
        cudaEvent_t comm_done;

        ncclComm_t comm;
        int64_t my_rank;
        int64_t my_size;

        cublasHandle_t my_handle;
};