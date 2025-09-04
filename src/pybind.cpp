#include "baseline_impl.h"
#include "overlap_impl.h"
#include "decomposition_impl.h"
#include "nccl_utils.h"
#include "rmsnorm/rmsnorm.h"

#include <torch/script.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

template<typename T>
void NcclInitWrapper(const c10::intrusive_ptr<T>& self, 
    const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id){
    self->NcclInit(tp_rank, tp_size, tp_id);
}

template<typename T>
void CublasInitWrapper(const c10::intrusive_ptr<T>& self){
    self->CublasInit();
}

template<typename T>
void CutlassGemmAllReduceWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){
    self->GemmAllReduce(A, B, C, Algo);
}

template<typename T>
void CutlassGemmReduceScatterWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo){
    self->GemmReduceScatter(A, B, C, D, Algo);
}

template<typename T>
void CutlassAll2AllWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo, at::Tensor mLen_CPU){
    self->GemmAll2All(A, B, C, D, Algo, mLen_CPU);
}

template<typename T>
void GemmAllReduceWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C){
    self->GemmAllReduce(A, B, C);
}

template<typename T>
void GemmReduceScatterWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D){
    self->GemmReduceScatter(A, B, C, D);
}

template<typename T>
void GemmAll2AllWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor mLen_CPU){
    self->GemmAll2All(A, B, C, D, mLen_CPU);
}

template<typename T>
void CutlassInitWrapper(const c10::intrusive_ptr<T>& self){
    self->CutlassInit();
}

template<typename T>
void GemmWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C){
    self->Gemm(A, B, C);
}

template<typename T>
void CutlassGemmWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){
    self->Gemm(A, B, C, Algo);
}

template<typename T>
void OverlapInitWrapper(const c10::intrusive_ptr<T>& self){
    self->OverlapInit();
}

template<typename T>
void AllReduceOverlapWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor MM, at::Tensor RA, int64_t rLDN, 
    at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t TilingAlgo, bool if_monitor){
    self->GemmAllReduceOverlap(A, B, C, MM, RA, rLDN, cSEG_CPU, cSEG_GPU, TilingAlgo, if_monitor);
}

template<typename T>
void ReduceScatterOverlapWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor MM, at::Tensor RA, at::Tensor RE, int64_t rLDN, 
    at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t TilingAlgo, bool if_monitor){
    self->GemmReduceScatterOverlap(A, B, C, D, MM, RA, RE, rLDN, cSEG_CPU, cSEG_GPU, TilingAlgo, if_monitor);
}

template<typename T>
void All2AllOverlapWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor MM, at::Tensor RA, int64_t rLDN, 
    at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, at::Tensor mLen_CPU, int64_t TilingAlgo){
    self->GemmAll2AllOverlap(A, B, C, D, MM, RA, rLDN, cSEG_CPU, cSEG_GPU, mLen_CPU, TilingAlgo);
}

template<typename T>
void NcclAllReduceWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C){
    self->NcclAllReduce(C);
}

template<typename T>
void SegAllReduceWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C,
    at::Tensor cSEG_CPU, int64_t SegNum){
    self->SegAllReduce(C, cSEG_CPU, SegNum);
}

template<typename T>
void NcclReduceScatterWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C){
    self->NcclReduceScatter(C);
}

template<typename T>
void NcclAll2AllWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C, at::Tensor D, at::Tensor mLen_CPU){
    self->NcclAll2All(C, D, mLen_CPU);
}

template<typename T>
void GemmReorderTileWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor MM, at::Tensor RA, int64_t rLDN, 
    at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t TilingAlgo){
    self->GemmReorderTile(A, B, C, MM, RA, rLDN, cSEG_CPU, cSEG_GPU, TilingAlgo);
}

template<typename T>
void GemmReorderTokenWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor MM, at::Tensor RA, at::Tensor RE, int64_t rLDN, 
    at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t TilingAlgo){
    self->GemmReorderToken(A, B, C, MM, RA, RE, rLDN, cSEG_CPU, cSEG_GPU, TilingAlgo);
}

TORCH_LIBRARY(flashoverlap_class, m) {

    // Class
    m.class_<BaselineImpl>("BaselineImpl")
        .def(torch::init())
        .def("nccl_init", &NcclInitWrapper<BaselineImpl>)
        .def("cublas_init", &CublasInitWrapper<BaselineImpl>)
        .def("gemm_allreduce", &GemmAllReduceWrapper<BaselineImpl>)
        .def("gemm_reducescatter", &GemmReduceScatterWrapper<BaselineImpl>)
        .def("gemm_all2all", &GemmAll2AllWrapper<BaselineImpl>)
        .def("cublas_gemm", &GemmWrapper<BaselineImpl>)
        .def("nccl_allreduce", &NcclAllReduceWrapper<BaselineImpl>)
        .def("nccl_reducescatter", &NcclReduceScatterWrapper<BaselineImpl>)
        .def("nccl_all2all", &NcclAll2AllWrapper<BaselineImpl>)
    ;
    m.class_<OverlapImpl>("OverlapImpl")
        .def(torch::init())
        .def("cutlass_init", &CutlassInitWrapper<OverlapImpl>)
        .def("cutlass_gemm", &CutlassGemmWrapper<OverlapImpl>)
        .def("gemm_allreduce_overlap", &AllReduceOverlapWrapper<OverlapImpl>)
        .def("gemm_reducescatter_overlap", &ReduceScatterOverlapWrapper<OverlapImpl>)
        .def("gemm_all2all_overlap", &All2AllOverlapWrapper<OverlapImpl>)
        .def("nccl_init", &NcclInitWrapper<OverlapImpl>)
        .def("gemm_allreduce", &CutlassGemmAllReduceWrapper<OverlapImpl>)
        .def("gemm_reducescatter", &CutlassGemmReduceScatterWrapper<OverlapImpl>)
        .def("gemm_all2all", &CutlassAll2AllWrapper<OverlapImpl>)
        .def("overlap_init", &OverlapInitWrapper<OverlapImpl>)
        .def("nccl_allreduce", &NcclAllReduceWrapper<OverlapImpl>)
        .def("nccl_reducescatter", &NcclReduceScatterWrapper<OverlapImpl>)
        .def("nccl_all2all", &NcclAll2AllWrapper<OverlapImpl>)
        .def("seg_allreduce", &SegAllReduceWrapper<OverlapImpl>)
        .def("gemm_reorder_tile", &GemmReorderTileWrapper<OverlapImpl>)
        .def("gemm_reorder_token", &GemmReorderTokenWrapper<OverlapImpl>)
    ;
    m.class_<DecompositionImpl>("DecompositionImpl")
        .def(torch::init())
        .def("nccl_init", &NcclInitWrapper<DecompositionImpl>)
        .def("cublas_init", &CublasInitWrapper<DecompositionImpl>)
        .def("gemm_allreduce", &GemmAllReduceWrapper<DecompositionImpl>)
        .def("gemm_reducescatter", &GemmReduceScatterWrapper<DecompositionImpl>)
        .def("gemm_all2all", &GemmAll2AllWrapper<DecompositionImpl>)
    ;
}

TORCH_LIBRARY(flashoverlap_op, m) {
    m.def("generate_nccl_id", &generate_nccl_id);
    m.def("reorder_rmsnorm", &reorder_rmsnorm);
    m.def("rmsnorm", &rmsnorm);
}