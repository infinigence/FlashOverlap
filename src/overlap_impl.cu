#include "nccl_utils.h"
#include "wait.cuh"

#include "tiling/gemm_tiling.cuh"
#include "tiling/signal_tiling.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

#include "overlap_impl.h"

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MAX_GROUP_SIZE 64

/// NIL Implementation: Overlap CUTLASS GEMM and NCCL AllReduce
OverlapImpl::OverlapImpl(){

}

OverlapImpl::~OverlapImpl(){

}

void OverlapImpl::CutlassInit(){
    this->gemm_stream = at::cuda::getCurrentCUDAStream().stream();
}

void OverlapImpl::Gemm(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    gemm_func_table[Algo](
        M, N, K, a_ptr, b_ptr, c_ptr, this->gemm_stream
    );
}

void OverlapImpl::NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id){

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

void OverlapImpl::GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    gemm_func_table[Algo](
        M, N, K, a_ptr, b_ptr, c_ptr, this->gemm_stream
    );

    NCCL_CHECK(ncclAllReduce((void *)c_ptr, (void *)c_ptr, (M * N), ncclFloat16, ncclSum, this->comm, this->gemm_stream));
}

void OverlapImpl::GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    gemm_func_table[Algo](
        M, N, K, a_ptr, b_ptr, c_ptr, this->gemm_stream
    );

    size_t recvcount = (M * N) / this->my_size;
    NCCL_CHECK(ncclReduceScatter((void *)c_ptr, (void *)(c_ptr + this->my_rank * recvcount), recvcount, 
        ncclFloat16, ncclSum, this->comm, this->gemm_stream));
}

void OverlapImpl::GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, 
    int64_t Algo, at::Tensor mLen_CPU){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());
    int* mlen_cpu_ptr = mLen_CPU.data_ptr<int>();

    gemm_func_table[Algo](
        M, N, K, a_ptr, b_ptr, c_ptr, this->gemm_stream
    );

    // Launch All2All after GEMM
    // First SEND
    int src_acc_addr = 0;
    // Then RECV
    int dst_acc_addr = 0;
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < this->my_size; i++){
        if (i == this->my_rank){continue;}
        size_t sendcount = mlen_cpu_ptr[this->my_rank * this->my_size + i] * N;
        NCCL_CHECK(ncclSend((void *)(c_ptr + src_acc_addr), sendcount, ncclFloat16, i, this->comm, this->gemm_stream));
        src_acc_addr += sendcount;

        size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N;
        NCCL_CHECK(ncclRecv((void *)(d_ptr + dst_acc_addr), recvcount, ncclFloat16, i, this->comm, this->gemm_stream));
        dst_acc_addr += recvcount;
    }
    NCCL_CHECK(ncclGroupEnd());
}

void OverlapImpl::OverlapInit(){
    cudaStreamCreateWithPriority(&this->comm_stream, cudaStreamNonBlocking, -5);
}

void OverlapImpl::NcclAllReduce(at::Tensor C){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    ncclAllReduce((void *)c_ptr, (void *)c_ptr, (M * N), ncclFloat16, ncclSum, this->comm, this->gemm_stream);
}

void OverlapImpl::SegAllReduce(at::Tensor C, at::Tensor cSEG_CPU, int64_t SegNum){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    int* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();

    int acc_addr = 0;
    for (int s = 0; s < SegNum; s++){
        int commSize = M * N / SegNum * cseg_cpu_ptr[s];
        NCCL_CHECK(ncclAllReduce((void *)(c_ptr + acc_addr), (void *)(c_ptr + acc_addr), commSize, ncclFloat16, ncclSum, this->comm, this->gemm_stream));
        acc_addr += commSize;
    }
}

void OverlapImpl::NcclReduceScatter(at::Tensor C){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    size_t recvcount = (M * N) / this->my_size;
    NCCL_CHECK(ncclReduceScatter((void *)c_ptr, (void *)(c_ptr + this->my_rank * recvcount), recvcount, 
        ncclFloat16, ncclSum, this->comm, this->gemm_stream));
}

void OverlapImpl::NcclAll2All(at::Tensor C, 
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
        NCCL_CHECK(ncclSend((void *)(c_ptr + src_acc_addr), sendcount, ncclFloat16, i, this->comm, this->gemm_stream));
        src_acc_addr += sendcount;

        size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N;
        NCCL_CHECK(ncclRecv((void *)(d_ptr + dst_acc_addr), recvcount, ncclFloat16, i, this->comm, this->gemm_stream));
        dst_acc_addr += recvcount;
    }
    NCCL_CHECK(ncclGroupEnd());
}

void OverlapImpl::GemmAllReduceOverlap(
        at::Tensor A,  // M, K
        at::Tensor B,  // N, K
        at::Tensor C,  // M, N
        at::Tensor MM, // TM + 1, TN
        at::Tensor RA, // TM, TN
        int64_t rLDN, 
        at::Tensor cSEG_CPU, // SegSize, how many communication segments
        at::Tensor cSEG_GPU, // SegSize, how many communication segments, on GPU
        int64_t Algo,
        bool if_monitor
        ){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    int TM = RA.size(0);
    int TN = RA.size(1);
    int TileNum = TM * TN;

    int SegSize = cSEG_GPU.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    int* mm_ptr = MM.data_ptr<int>();
    int* ra_ptr = RA.data_ptr<int>();

    int* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();
    int* cseg_gpu_ptr = cSEG_GPU.data_ptr<int>();

    int acc_addr = 0;
    signal_func_table[Algo](
        M, N, K, rLDN, cseg_gpu_ptr, a_ptr, b_ptr, c_ptr, mm_ptr, ra_ptr, if_monitor, this->gemm_stream
    );
    // for (int iter = 0; iter < SegSize; iter++){
    //     int this_seg = cseg_cpu_ptr[iter];
    //     int commSize = M * N / TileNum * this_seg;
    //     // The signal is reset by the wait kernel
    //     kernel_wait_flag<<<1, 1, 0, this->comm_stream>>> (this_seg, (mm_ptr + iter));
    //     // Communicate the data
    //     NCCL_CHECK(ncclAllReduce((void *)(c_ptr + acc_addr), (void *)(c_ptr + acc_addr), commSize, ncclFloat16, ncclSum, this->comm, this->comm_stream));
    //     acc_addr += commSize;
    // }

    // cudaEventCreateWithFlags(&this->gemm_finished, cudaEventDisableTiming);
    // cudaEventRecord(this->gemm_finished, this->comm_stream);
    // cudaStreamWaitEvent(this->gemm_stream, this->gemm_finished, 0);
    // cudaEventDestroy(this->gemm_finished);
}
