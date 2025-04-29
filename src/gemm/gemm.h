#include <cuda_fp16.h>

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, 
int InstructionM, int InstructionN, int InstructionK, int NumStages, int SwizzleSize, int SplitK>
void cutlass_gemm_splitk(int M, int N, int K, const half* A, const half* B, half* D, cudaStream_t stream);