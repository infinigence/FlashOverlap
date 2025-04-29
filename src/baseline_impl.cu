#include "nccl_utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

#include "baseline_impl.h"

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MAX_GROUP_SIZE 16

/// Baseline Implementation: cuBLAS for GEMM and NCCL for AllReduce
BaselineImpl::BaselineImpl(){
    cublasCreate(&this->my_handle);
    this->my_rank = 0;
    this->my_size = 1;
}

BaselineImpl::~BaselineImpl(){
    cublasDestroy(this->my_handle);
    // ncclCommDestroy(this->comm);
}

void BaselineImpl::GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C){

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

    // Launch GEMM
    cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,
                    (const void*)reinterpret_cast<half *>(&alpha_half),
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Launch AllReduce after GEMM
    NCCL_CHECK(ncclAllReduce((void *)c_ptr, (void *)c_ptr, (M * N), ncclFloat16, ncclSum, this->comm, this->my_stream));
}

void BaselineImpl::GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C){

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

    // Launch GEMM
    cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,
                    (const void*)reinterpret_cast<half *>(&alpha_half),
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Launch AllReduce after GEMM
    size_t recvcount = (M * N) / this->my_size;
    NCCL_CHECK(ncclReduceScatter((void *)c_ptr, (void *)(c_ptr + this->my_rank * recvcount), recvcount, 
        ncclFloat16, ncclSum, this->comm, this->my_stream));
}

void BaselineImpl::GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C,
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

    // Launch GEMM
    cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,
                    (const void*)reinterpret_cast<half *>(&alpha_half),
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Launch All2All after GEMM
    // First SEND
    int src_acc_addr = 0;
    // Then RECV
    int dst_acc_addr = 0;
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < this->my_size; i++){
        if (i == this->my_rank){continue;}
        size_t sendcount = mlen_cpu_ptr[this->my_rank * this->my_size + i] * N;
        NCCL_CHECK(ncclSend((void *)(c_ptr + src_acc_addr), sendcount, ncclFloat16, i, this->comm, this->my_stream));
        src_acc_addr += sendcount;

        size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N;
        NCCL_CHECK(ncclRecv((void *)(d_ptr + dst_acc_addr), recvcount, ncclFloat16, i, this->comm, this->my_stream));
        dst_acc_addr += recvcount;
    }
    NCCL_CHECK(ncclGroupEnd());
}

void BaselineImpl::NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id)
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

void BaselineImpl::CublasInit(){

    // prepare for GEMM
    this->my_stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(this->my_handle, this->my_stream);
    cublasSetMathMode(this->my_handle, CUBLAS_TENSOR_OP_MATH);
}

void BaselineImpl::Gemm(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH);

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,
                    (const void*)reinterpret_cast<half *>(&alpha_half), 
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K,  
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void BaselineImpl::NcclAllReduce(at::Tensor C){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    ncclAllReduce((void *)c_ptr, (void *)c_ptr, (M * N), ncclFloat16, ncclSum, this->comm, this->my_stream);
}

void BaselineImpl::NcclReduceScatter(at::Tensor C){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    size_t recvcount = (M * N) / this->my_size;
    NCCL_CHECK(ncclReduceScatter((void *)c_ptr, (void *)(c_ptr + this->my_rank * recvcount), recvcount, 
        ncclFloat16, ncclSum, this->comm, this->my_stream));
}

void BaselineImpl::NcclAll2All(at::Tensor C, 
    at::Tensor D, // [world_size - 1, M, N]
    at::Tensor mLen_CPU // [world_size, world_size]
    ){
    
    int M = C.size(0);
    int N = C.size(1);

    assert(mLen_CPU.size(0) == this->my_size);
    assert(mLen_CPU.size(1) == this->my_size);

    int* mlen_cpu_ptr = mLen_CPU.data_ptr<int>();
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());

    // First SEND
    int src_acc_addr = 0;
    // Then RECV
    int dst_acc_addr = 0;
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < this->my_size; i++){
        if (i == this->my_rank){continue;}
        size_t sendcount = mlen_cpu_ptr[this->my_rank * this->my_size + i] * N;
        NCCL_CHECK(ncclSend((void *)(c_ptr + src_acc_addr), sendcount, ncclFloat16, i, this->comm, this->my_stream));
        src_acc_addr += sendcount;

        size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N;
        NCCL_CHECK(ncclRecv((void *)(d_ptr + dst_acc_addr), recvcount, ncclFloat16, i, this->comm, this->my_stream));
        dst_acc_addr += recvcount;
    }
    NCCL_CHECK(ncclGroupEnd());
}
