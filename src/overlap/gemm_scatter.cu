#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/epilogue/thread/scale_type.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/fast_math.h"

#include "gemm_scatter.h"
#include "gemm_with_scatter.h"

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, 
int InstructionM, int InstructionN, int InstructionK, int NumStages, int SwizzleSize, int SplitK>
void cutlass_gemm_scatter(int M, int N, int K, int ReLDN, int* CommThr, half* A, half* B, half* D, int* MM, int* RA, int* RE, bool Monitor, cudaStream_t stream = nullptr){

    using ThreadblockShape = cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
    using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
    using InstructionShape = cutlass::gemm::GemmShape<InstructionM, InstructionN, InstructionK>;
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // A matrix configuration
    using ElementA = cutlass::half_t;                                // Element type for A matrix operand
    using LayoutA = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = cutlass::half_t;                                // Element type for B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;                      // Layout type for B matrix operand
    constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using ElementC = cutlass::half_t;                                // Element type for C and D matrix operands
    using LayoutC = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
    constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)

    // Multiply-accumulate blocking/pipelining details
    using ElementAccumulator  = cutlass::half_t;                          // Element type for internal accumulation
    using ArchTag             = cutlass::arch::Sm80;                      // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass       = cutlass::arch::OpClassTensorOp;           // Operator class tag

    // Epilogue output operator
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,               // Element type for C and D matrix operands
        AlignmentC,             // Memory access granularity of C and D matrix in units of elements
        ElementAccumulator,     // Element type from internal accumaccumulation
        ElementAccumulator>;    // Data type used to compute linear combination

    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleSize>;

    static bool const kInternalTranspose = cutlass::platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value;

    using GemmScatter = cutlass::GemmScatter<
      ElementA,
      LayoutA, 
      ElementB,
      LayoutB, 
      ElementC,
      LayoutC,
      ElementAccumulator,
      EpilogueOp,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      NumStages,
      SwizzleSize
    >;

    typename GemmScatter::Arguments arguments(
      problem_size,
      reinterpret_cast<cutlass::half_t*>(A),
      reinterpret_cast<cutlass::half_t*>(B), 
      reinterpret_cast<cutlass::half_t*>(D), 
      reinterpret_cast<cutlass::half_t*>(D), 
      (int64_t)K, 
      (int64_t)K, 
      (int64_t)N, 
      (int64_t)N, 
      {
        ElementAccumulator(1.0f),
        ElementAccumulator(0.0f)
      },
      MM, 
      RA, 
      (N / ThreadblockN),
      ReLDN,
      CommThr, 
      RE, 
      Monitor
    );

    GemmScatter gemm_op;

    CUTLASS_CHECK(gemm_op.initialize(arguments));
    CUTLASS_CHECK(gemm_op(stream));
}

#define CUTLASS_GEMM_SCATTER_INIT(ThreadblockM, ThreadblockN, ThreadblockK, WarpM,         \
                                        WarpN, WarpK, InstructionM, InstructionN,                \
                                        InstructionK, NumStages, SwizzleSize, SplitK)                                                           \
    template void                                                                                                    \
    cutlass_gemm_scatter<ThreadblockM, ThreadblockN, ThreadblockK, WarpM, WarpN, \
                                    WarpK, InstructionM, InstructionN, InstructionK, NumStages, SwizzleSize, SplitK>(            \
        int M, int N, int K, int ReLDN, int* CommThr, half* A, half* B, half* D, int* MM, int* RA, int* RE, bool Monitor, cudaStream_t stream = nullptr)

#include "../inc/scatter_instances.inc"

#undef CUTLASS_GEMM_SCATTER_INIT