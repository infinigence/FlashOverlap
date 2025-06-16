#pragma once

#include <nccl.h>
#include <vector>
#include <cublas_v2.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/util/device_memory.h"

class OverlapImpl : public torch::CustomClassHolder {
    public:
        OverlapImpl();
        ~OverlapImpl();

        void CutlassInit();
        void NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id);
        void OverlapInit();

        void Gemm(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo);

        void GemmAllReduceOverlap(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor MM, at::Tensor RA, int64_t rLDN, at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t Algo, bool if_monitor);
        void GemmReduceScatterOverlap(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor MM, at::Tensor RA, at::Tensor RE, int64_t rLDN, at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t Algo, bool if_monitor);

        void GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo);
        void GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo);
        void GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo, at::Tensor mLen_CPU);

        void SegAllReduce(at::Tensor C, at::Tensor cSEG_CPU, int64_t SegNum);
        void NcclAllReduce(at::Tensor C);
        void NcclReduceScatter(at::Tensor C);
        void NcclAll2All(at::Tensor C, at::Tensor D, at::Tensor mLen_CPU);

    private:
        cudaStream_t gemm_stream;
        cudaStream_t comm_stream;
        cudaEvent_t gemm_finished;

        ncclComm_t comm;
        int64_t my_rank;
        int64_t my_size;
};