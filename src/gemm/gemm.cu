#include <cuda_fp16.h>

#include "gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/tensor_view_io.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, 
int InstructionM, int InstructionN, int InstructionK, int NumStages, int SwizzleSize, int SplitK>
void cutlass_gemm_splitk(int M, int N, int K, const half* A, const half* B, half* D, cudaStream_t stream = nullptr){

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

    // Classic data-parallel device GEMM implementation type
    using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleSize>,
        NumStages,
        AlignmentA,
        AlignmentB>;

    int64_t batch_stride_A = problem_size.m() * problem_size.k();
    int64_t batch_stride_B = problem_size.k() * problem_size.n();
    int64_t batch_stride_C = problem_size.m() * problem_size.n();
    int64_t batch_stride_D = problem_size.m() * problem_size.n();

    auto arguments = typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                             // problem_size
        SplitK,                                        // batch count / splitk slices
        {                                         // epilogue parameters
        ElementAccumulator(1.0f),
        ElementAccumulator(0.0f)
        },
        A,                   // ptr_A
        B,                   // ptr_B
        D,                   // ptr_C
        D,                   // ptr_D
        batch_stride_A,      // batch_stride_A
        batch_stride_B,      // batch_stride_B
        batch_stride_C,      // batch_stride_C
        batch_stride_D,      // batch_stride_D
        K,              // stride_a
        K,              // stride_b
        N,              // stride_c
        N);                 // stride_d                                  

    DeviceGemmBasic gemm_op;
    // workspace_size = 0
    size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    CUTLASS_CHECK(gemm_op(stream));
}

#define CUTLASS_GEMM_SPLITK_INIT(ThreadblockM, ThreadblockN, ThreadblockK, WarpM,         \
                                        WarpN, WarpK, InstructionM, InstructionN,                \
                                        InstructionK, NumStages, SwizzleSize, SplitK)                                                           \
    template void                                                                                                    \
    cutlass_gemm_splitk<ThreadblockM, ThreadblockN, ThreadblockK, WarpM, WarpN, \
                                    WarpK, InstructionM, InstructionN, InstructionK, NumStages, SwizzleSize, SplitK>(            \
        int M, int N, int K, const half* A, const half* B, half* D, cudaStream_t stream = nullptr)

#include "../inc/gemm_instances.inc"

#undef CUTLASS_GEMM_SPLITK_INIT