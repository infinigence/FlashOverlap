#include "nccl_utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

#include "decomposition_impl.h"

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MAX_GROUP_SIZE 16
#define ITER_NUM 2

/// Baseline Implementation: cuBLAS for GEMM and NCCL for AllReduce
DecompositionImpl::DecompositionImpl(){
    this->my_rank = 0;
    this->my_size = 1;

    cudaStreamCreateWithPriority(&this->comm_stream, cudaStreamNonBlocking, -5);
}

DecompositionImpl::~DecompositionImpl(){
    cublasDestroy(this->my_handle);
    // ncclCommDestroy(this->comm);
}

void DecompositionImpl::NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id)
{
    this->my_rank = tp_rank;
    this->my_size = tp_size;
    
    ncclUniqueId tp_uid;
    memcpy(tp_uid.internal, &tp_id[0], NCCL_UNIQUE_ID_BYTES);

    if (this->my_size == 1) {
        this->comm = nullptr;
        return;
    }
    NCCL_CHECK(ncclCommInitRank(&this->comm, this->my_size, tp_uid, this->my_rank));
}

void DecompositionImpl::CublasInit(){
    // prepare for GEMM
    cublasCreate(&this->my_handle);
    this->gemm_stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(this->my_handle, this->gemm_stream);
    cublasSetMathMode(this->my_handle, CUBLAS_TENSOR_OP_MATH);
}

void DecompositionImpl::GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C){

    // Check if NCCL is initilized
    if (this->comm == nullptr) {
        return;
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    // prepare for NCCL
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    cudaEvent_t gemm_done[ITER_NUM];
    cudaEventCreateWithFlags(&this->comm_done, cudaEventDisableTiming);

    for(int i = 0; i < ITER_NUM; i++){
        cudaEventCreateWithFlags(&gemm_done[i], cudaEventDisableTiming);
        // Launch GEMM
        cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                        N, M / ITER_NUM, K,
                        // (const void*)reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()), 
                        (const void*)reinterpret_cast<half *>(&alpha_half),
                        // (const void*)(&alpha), 
                        (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                        CUDA_R_16F, K, 
                        (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>() + M / ITER_NUM * K * i), 
                        CUDA_R_16F, K, 
                        // (const void*)reinterpret_cast<half *>(beta_half.data_ptr<at::Half>()), 
                        (const void*)reinterpret_cast<half *>(&beta_half),
                        // (const void*)(&beta), 
                        (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>() + M / ITER_NUM * N * i), 
                        CUDA_R_16F, N,
                        CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        cudaEventRecord(gemm_done[i], this->gemm_stream);
        cudaStreamWaitEvent(this->comm_stream, gemm_done[i], 0);

        // Launch AllReduce after GEMM
        NCCL_CHECK(ncclAllReduce((void *)(c_ptr + M / ITER_NUM * N * i), (void *)(c_ptr + M / ITER_NUM * N * i), (M / ITER_NUM * N), 
            ncclFloat16, ncclSum, this->comm, this->comm_stream));
    }

    for(int j = 0; j < ITER_NUM; j++){
        cudaEventDestroy(gemm_done[j]);
    }
    cudaEventRecord(this->comm_done, this->comm_stream);
    cudaStreamWaitEvent(this->gemm_stream, this->comm_done, 0);
    cudaEventDestroy(this->comm_done);
}

void DecompositionImpl::GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D){

    // Check if NCCL is initilized
    if (this->comm == nullptr) {
        return;
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    // prepare for NCCL
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());

    cudaEvent_t gemm_done[ITER_NUM];
    cudaEventCreateWithFlags(&this->comm_done, cudaEventDisableTiming);

    for (int i = 0; i < ITER_NUM; i++){
        cudaEventCreateWithFlags(&gemm_done[i], cudaEventDisableTiming);
        // Launch GEMM
        cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                        N, M / ITER_NUM, K,
                        // (const void*)reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()), 
                        (const void*)reinterpret_cast<half *>(&alpha_half),
                        // (const void*)(&alpha), 
                        (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                        CUDA_R_16F, K, 
                        (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>() + M / ITER_NUM * K * i), 
                        CUDA_R_16F, K, 
                        // (const void*)reinterpret_cast<half *>(beta_half.data_ptr<at::Half>()), 
                        (const void*)reinterpret_cast<half *>(&beta_half),
                        // (const void*)(&beta), 
                        (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>() + M / ITER_NUM * N * i), 
                        CUDA_R_16F, N,
                        CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        cudaEventRecord(gemm_done[i], this->gemm_stream);
        cudaStreamWaitEvent(this->comm_stream, gemm_done[i], 0);

        // Launch ReduceScatter after GEMM
        size_t recvcount = (M / ITER_NUM * N) / this->my_size;
        NCCL_CHECK(ncclReduceScatter((void *)(c_ptr + M / ITER_NUM * N * i), 
            (void *)(d_ptr + M / ITER_NUM * N * i + this->my_rank * recvcount), 
            recvcount, ncclFloat16, ncclSum, this->comm, this->comm_stream));
    }

    for(int j = 0; j < ITER_NUM; j++){
        cudaEventDestroy(gemm_done[j]);
    }
    cudaEventRecord(this->comm_done, this->comm_stream);
    cudaStreamWaitEvent(this->gemm_stream, this->comm_done, 0);
    cudaEventDestroy(this->comm_done);
}

// TODO: a transpose needed
void DecompositionImpl::GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C,
    at::Tensor D, at::Tensor mLen_CPU){

    // Check if NCCL is initilized
    if (this->comm == nullptr) {
        return;
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    assert(mLen_CPU.size(0) == this->my_size);
    assert(mLen_CPU.size(1) == this->my_size);

    int* mlen_cpu_ptr = mLen_CPU.data_ptr<int>();

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    // prepare for NCCL
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());

    cudaEvent_t gemm_done[ITER_NUM];
    cudaEventCreateWithFlags(&this->comm_done, cudaEventDisableTiming);

    int src_acc_addr = 0;
    int dst_acc_addr = 0;
    for (int j = 0; j < ITER_NUM; j++){
        cudaEventCreateWithFlags(&gemm_done[j], cudaEventDisableTiming);
        // Launch GEMM
        cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                        N / ITER_NUM, M, K,
                        // (const void*)reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()), 
                        (const void*)reinterpret_cast<half *>(&alpha_half),
                        // (const void*)(&alpha), 
                        (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>() + N / ITER_NUM * K * j),
                        CUDA_R_16F, K, 
                        (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                        CUDA_R_16F, K, 
                        // (const void*)reinterpret_cast<half *>(beta_half.data_ptr<at::Half>()), 
                        (const void*)reinterpret_cast<half *>(&beta_half),
                        // (const void*)(&beta), 
                        (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>() + N / ITER_NUM * M * j), 
                        CUDA_R_16F, N / ITER_NUM,
                        CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        cudaEventRecord(gemm_done[j], this->gemm_stream);
        cudaStreamWaitEvent(this->comm_stream, gemm_done[j], 0);

        // Launch All2All after GEMM
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < this->my_size; i++){
            if (i == this->my_rank){continue;}
            size_t sendcount = mlen_cpu_ptr[this->my_rank * this->my_size + i] * N / ITER_NUM;
            NCCL_CHECK(ncclSend((void *)(c_ptr + src_acc_addr), sendcount, ncclFloat16, i, this->comm, this->comm_stream));
            src_acc_addr += sendcount;

            size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N / ITER_NUM;
            NCCL_CHECK(ncclRecv((void *)(d_ptr + dst_acc_addr), recvcount, ncclFloat16, i, this->comm, this->comm_stream));
            dst_acc_addr += recvcount;
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    for(int j = 0; j < ITER_NUM; j++){
        cudaEventDestroy(gemm_done[j]);
    }
    cudaEventRecord(this->comm_done, this->comm_stream);
    cudaStreamWaitEvent(this->gemm_stream, this->comm_done, 0);
    cudaEventDestroy(this->comm_done);
}