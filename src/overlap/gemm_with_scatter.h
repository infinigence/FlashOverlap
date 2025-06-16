/***************************************************************************************************
 * Adapted from NVIDIA example
 * Authored by hkeee.
 **************************************************************************************************/

/*! \file
    \brief A file contains all functioning classes needed by monitoring the threadblock-tile mappings.

*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

#include "gemm_with_epilogue_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }
// we don't want that
// /////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

template <
  typename ThreadblockShape_,
  int ThreadCount,
  typename OutputTileIterator_,
  typename AccumulatorTile_,
  typename ElementAccumulator_,
  typename ElementwiseFunctor_
>
class EpilogueVisitorScatter {
public:

  using AccumulatorTile = AccumulatorTile_;

  using ThreadblockShape   = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static int const kRowIterations = OutputTileIterator::ThreadMap::Iterations::kRow;

  static int const kThreads = OutputTileIterator::ThreadMap::kThreads;

  using ElementOutput = typename OutputTileIterator::Element;

  static int const kDeltaRow = OutputTileIterator::ThreadMap::Delta::kRow;

  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementAccumulator = ElementAccumulator_;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  // using LayernormFragment = Array<ElementLayernormCompute, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;
  using TensorRefD = TensorRef<ElementOutput, LayoutOutput>;

  static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::RowArrangement::Detail::kShapeWidth;
  static int const kThreadsInColumn = kThreads / kThreadsPerRow;
  static int const kHalfThreadsPerRow = (kThreadsPerRow >> 1);

  static int const ThreadblockM = ThreadblockShape::kM;
  static int const ThreadblockN = ThreadblockShape::kN;

  /// Argument structure
  struct Arguments {

    typename ElementwiseFunctor::Params   elementwise;
    TensorRefD                            ref_C;
    TensorRefD                            ref_D;
    TensorRefD                            ref_ND;
    int                                   *ptr_Monitored_Matrix;
    int                                   *ptr_Reorder_Array;
    int                                    kMonitoredColmun;
    int                                    kReorderedColumn;  
    int                                   *kCommu_Seg_Array;
    int                                   *ptr_Row_Array;
    bool                                  if_monitor;
    
    //
    // Methods
    //
    Arguments() { }

    Arguments(
      typename ElementwiseFunctor::Params   elementwise_,
      TensorRefD                            ref_C_,
      TensorRefD                            ref_D_,
      TensorRefD                            ref_ND_,
      int                                  *ptr_Monitored_Matrix_,
      int                                  *ptr_Reorder_Array_, 
      int                                   kMonitoredColmun_,
      int                                   kReorderedColumn_,
      int                                  *kCommu_Seg_Array_,
      int                                  *ptr_Row_Array_, 
      bool                                  if_monitor
    ):
      elementwise(elementwise_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      ref_ND(ref_ND_), 
      ptr_Monitored_Matrix(ptr_Monitored_Matrix_),
      ptr_Reorder_Array(ptr_Reorder_Array_),
      kMonitoredColmun(kMonitoredColmun_),
      kReorderedColumn(kReorderedColumn_),
      kCommu_Seg_Array(kCommu_Seg_Array_),
      ptr_Row_Array(ptr_Row_Array_), 
      if_monitor(if_monitor)
    {

    }
  };

  struct Params {

    typename ElementwiseFunctor::Params   elementwise;
    typename OutputTileIterator::Params   params_C;
    typename OutputTileIterator::Params   params_D;
    typename OutputTileIterator::Params   params_ND;
    typename OutputTileIterator::Element *ptr_C;
    typename OutputTileIterator::Element *ptr_D;
    int                                  *ptr_Monitored_Matrix;
    int                                  *ptr_Reorder_Array;
    int                                   kMonitoredColmun;
    int                                   kReorderedColumn;
    int                                  *kCommu_Seg_Array;
    int                                  *ptr_Row_Array;
    bool                                  if_monitor;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params():
      ptr_D(nullptr)
    {

    }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args):
      elementwise(args.elementwise),
      params_C(args.ref_C.layout()),
      params_D(args.ref_D.layout()),
      params_ND(args.ref_ND.layout()),
      ptr_C(args.ref_C.data()),
      ptr_D(args.ref_D.data()),
      ptr_Monitored_Matrix(args.ptr_Monitored_Matrix),
      ptr_Reorder_Array(args.ptr_Reorder_Array), 
      kMonitoredColmun(args.kMonitoredColmun),
      kReorderedColumn(args.kReorderedColumn),
      kCommu_Seg_Array(args.kCommu_Seg_Array),
      ptr_Row_Array(args.ptr_Row_Array), 
      if_monitor(args.if_monitor)
    {

    }
  };

  /// Shared storage
  struct SharedStorage {

  };

private:

  Params const &                        params_;
  SharedStorage &                       shared_storage_;
  MatrixCoord                           extent_;
  ElementwiseFunctor                    elementwise_;

  OutputTileIterator                    iterator_C_;
  OutputTileIterator                    iterator_D_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator                    alpha_;
  ElementAccumulator                    beta_;
  
  MatrixCoord                           thread_offset_;
  MatrixCoord                           threadblock_offset_;

public:

  CUTLASS_DEVICE
  EpilogueVisitorScatter(
    Params const &params,                                         ///< Parameters routed to the epilogue
    SharedStorage &shared_storage,                                ///< Shared storage needed by the functors here
    MatrixCoord const &problem_size,                              ///< Problem size of the output
    int thread_idx,                                               ///< Thread index within the threadblock
    int warp_idx,                                                 ///< Warp index within the threadblock
    int lane_idx,                                                 ///< Lane index within the warp
    MatrixCoord const &threadblock_offset = MatrixCoord(0, 0)
  ):
    params_(params),
    shared_storage_(shared_storage),
    extent_(reshape(problem_size)),
    elementwise_(params.elementwise),
    iterator_C_(params.params_C, params.ptr_C, problem_size, thread_idx, threadblock_offset),
    iterator_D_(params.params_ND, params.ptr_D, reshape(problem_size), thread_idx, map_to_d(threadblock_offset), params.ptr_Row_Array),
    threadblock_offset_(map_to_d(threadblock_offset))
  {
    alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr : params.elementwise.alpha);
    beta_ =  (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr : params.elementwise.beta);

    if (beta_ == ElementAccumulator()) {
      iterator_C_.clear_mask();
    }
  }

  CUTLASS_DEVICE
  MatrixCoord reshape(
    MatrixCoord const &problem_size
  ){
      int row_size = problem_size.row();
      int col_size = problem_size.column();

      return MatrixCoord(row_size * col_size / (params_.kReorderedColumn * ThreadblockN), 
        params_.kReorderedColumn * ThreadblockN);
  }

  CUTLASS_DEVICE
  MatrixCoord map_to_d(
    MatrixCoord const &threadblock_offset
  ){
      int tile_idx = (threadblock_offset.row() / ThreadblockM) * params_.kMonitoredColmun + \
      threadblock_offset.column() / ThreadblockN;
      int reordered_tile_idx = params_.ptr_Reorder_Array[tile_idx];
      
      return MatrixCoord((reordered_tile_idx / params_.kReorderedColumn) * ThreadblockM, \
        (reordered_tile_idx % params_.kReorderedColumn) * ThreadblockN);
  }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(
    int split_k_index,                                            ///< Index of this threadblock within split-K partitioned scheme
    int split_k_slices) {                                         ///< Total number of split-K slices

  }

  /// Called to set the batch index
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {

  }

  /// Called at the start of the epilogue just before iterating over accumulator slices
  CUTLASS_DEVICE
  void begin_epilogue() {

  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {

  }

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(
    int iter_idx,
    int row_idx,
    int column_idx,
    int frag_idx,
    AccumulatorFragment const &accum) {

    AccumulatorFragment result;

    thread_offset_ =
      iterator_D_.thread_start() +
      OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    NumericArrayConverter<ElementAccumulator, ElementOutput, kElementsPerAccess> source_converter;
    // OutputVector &source_vector = reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];

    // bool column_guard = (thread_offset_.column() < extent_.column());

    // if (elementwise_.kScale == cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
    result = source_converter(elementwise_(accum));
    // }else{
    // result = source_converter(elementwise_(accum, source_vector));
    // }

    // Convert to the output
    NumericArrayConverter<ElementOutput, ElementAccumulator, kElementsPerAccess> output_converter;
    OutputVector &output = reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
    output = output_converter(result);
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void end_row(int row_idx) {

  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {

    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {

    if (threadIdx.x > 0) {return;}
    int tile_idx = (threadblock_offset_.row() / ThreadblockM) * params_.kReorderedColumn + \
      threadblock_offset_.column() / ThreadblockN;

    int idx_bound = params_.kCommu_Seg_Array[0];
    int iter_idx = 0;
    while(idx_bound <= tile_idx){
      iter_idx += 1;
      idx_bound += params_.kCommu_Seg_Array[iter_idx];
    }

    int local_order = atomicAdd(&params_.ptr_Monitored_Matrix[iter_idx], 1);

    if (params_.if_monitor){
      int global_order = atomicAdd(&params_.ptr_Monitored_Matrix[(params_.kMonitoredColmun - 1)], 1);

      arch::global_store<int, sizeof(int)>(
                global_order,
                (void *)(params_.ptr_Monitored_Matrix + params_.kMonitoredColmun + tile_idx),
                true);
    }
  }

private:

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

///
template <
  typename ElementInputA_,
  typename LayoutInputA_, 
  typename ElementInputB_,
  typename LayoutInputB_, 
  typename ElementOutput_,
  typename LayoutOutput_,
  typename ElementCompute_,
  typename EpilogueFunctorOp_,
  typename ThreadblockShape_,
  typename WarpShape_,
  typename InstructionShape_,
  int Stages,
  int SwizzleSize
>
class GemmScatter {
public:

  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  static bool const kInternalTranspose = cutlass::platform::is_same<LayoutOutput_, cutlass::layout::ColumnMajor>::value;
 
  using OperatorClass       = cutlass::arch::OpClassTensorOp;
  using ArchTag             = cutlass::arch::Sm80;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  using LayoutInputA = LayoutInputA_;
  using LayoutInputB = LayoutInputB_;
  using LayoutOutputC = LayoutOutput_;

  using ElementInputA = ElementInputA_;
  using ElementInputB = ElementInputB_;
  using ElementOutputC = ElementOutput_;
  using ElementCompute = ElementCompute_;

  using EpilogueFunctorOp = EpilogueFunctorOp_;

  using TensorRefA = TensorRef<ElementInputA, LayoutInputA>;
  using TensorRefB = TensorRef<ElementInputB, LayoutInputB>;
  using TensorRefC = TensorRef<ElementOutputC, LayoutOutputC>;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape        = WarpShape_;
  using InstructionShape = InstructionShape_;

  static int const kStages = Stages;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleSize>;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  using MapArguments = cutlass::gemm::kernel::detail::MapArguments<
    ElementInputA,
    LayoutInputA,
    cutlass::ComplexTransform::kNone,
    128 / cutlass::sizeof_bits<ElementInputA>::value,
    ElementInputB,
    LayoutInputB,
    cutlass::ComplexTransform::kNone,
    128 / cutlass::sizeof_bits<ElementInputB>::value,
    LayoutOutputC,
    kInternalTranspose
  >;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementOutputC,
    typename MapArguments::LayoutC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    SwizzleThreadBlock,
    kStages,
    true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementInputA, ElementInputB, ElementOutputC, ElementCompute>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    false, 
    false, 
    true
  >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Epilogue visitor
  using EpilogueVisitor = kernel::EpilogueVisitorScatter<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    typename DefaultGemmKernel::Epilogue::AccumulatorFragmentIterator::AccumulatorTile,
    ElementCompute,
    EpilogueFunctorOp
  >;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    EpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
  >::Epilogue;

  // GEMM
  using GemmEpilogueFusion = gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    SwizzleThreadBlock // does it mean we need an extra sreamk impl. ?
  >;


public:

  /// Arguments class
  struct Arguments {

    typename GemmEpilogueFusion::Arguments         gemm;

    //
    // Methods
    //
    Arguments() { }

    Arguments(
      cutlass::gemm::GemmCoord problem_size,
      ElementInputA * ptr_A,
      ElementInputB * ptr_B,
      ElementOutputC * ptr_C,
      ElementOutputC * ptr_D,
      int64_t    ldm_A,
      int64_t    ldm_B,
      int64_t    ldm_C,
      int64_t    ldm_D,
      typename EpilogueFunctorOp::Params linear_scaling, 
      int * ptr_MM, 
      int * ptr_RA, 
      int ldm_MM,
      int red_TN,
      int * thr_CM,
      int * ptr_RE, 
      bool Monitor
    ):
      gemm(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {kInternalTranspose ? problem_size.n() : problem_size.m(),\
         kInternalTranspose ? problem_size.m() : problem_size.n(),\
         problem_size.k()},
        {kInternalTranspose ? ptr_B : ptr_A, \
        kInternalTranspose ? ldm_B : ldm_A},
        {kInternalTranspose ? ptr_A : ptr_B, \
        kInternalTranspose ? ldm_A : ldm_B},
        typename EpilogueVisitor::Arguments(
          linear_scaling,
          {ptr_C, ldm_C},
          {ptr_D, ldm_D},
          {ptr_D, red_TN * problem_size.n()/ldm_MM},
          ptr_MM,
          ptr_RA, 
          ldm_MM,
          red_TN, 
          thr_CM,
          ptr_RE, 
          Monitor
        )
      )
    {

    }
  };

  struct Params {

    typename GemmEpilogueFusion::Params         gemm;
   
    // Methods
    //
    Params() { }

    Params(Arguments const &args):
      gemm(args.gemm)
    {

    }
  };

public:

  // Gemm


  //
  // Methods
  //

private:

  Params params_;

public:

  /// Ctor
  GemmScatter() {

  }

  // Initialize
  Status initialize(Arguments const &args) {

    params_ = Params(args);

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {

    //
    // Launch the fused kernel
    //

    dim3 gemm_grid = SwizzleThreadBlock().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(GemmEpilogueFusion::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename GemmEpilogueFusion::SharedStorage));

    cudaError_t result;

    if (gemm_smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<GemmEpilogueFusion>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    gemm_smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmEpilogueFusion><<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////