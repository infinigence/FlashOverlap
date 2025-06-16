#include <cuda_fp16.h>

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, 
int InstructionM, int InstructionN, int InstructionK, int NumStages, int SwizzleSize, int SplitK>
void cutlass_gemm_scatter(int M, int N, int K, int ReLDN, int* CommThr, half* A, half* B, half* D, int* MM, int* RA, int* RE, bool Monitor, cudaStream_t stream);