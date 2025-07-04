#include "gemm_dispatcher.h"

ScatterFuncPtr scatter_func_table[] = {
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 8, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 4, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 4, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 4, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 4, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 4, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 4, 8, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 5, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 5, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 5, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 5, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 5, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 32, 64, 64, 32, 16, 8, 16, 5, 8, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 3, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 3, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 3, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 3, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 3, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 3, 8, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 4, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 4, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 4, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 4, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 4, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 4, 8, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 5, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 5, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 5, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 5, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 5, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 32, 16, 8, 16, 5, 8, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 3, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 3, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 3, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 3, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 3, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 3, 8, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 4, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 4, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 4, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 4, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 4, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 4, 8, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 5, 1, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 5, 2, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 5, 3, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 5, 4, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 5, 6, 1>,
    &cutlass_gemm_scatter<128, 128, 64, 64, 64, 64, 16, 8, 16, 5, 8, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, 1, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, 2, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, 3, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, 4, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, 6, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, 8, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 4, 1, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 4, 2, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 4, 3, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 4, 4, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 4, 6, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 4, 8, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 5, 1, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 5, 2, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 5, 3, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 5, 4, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 5, 6, 1>,
    &cutlass_gemm_scatter<128, 256, 32, 64, 64, 32, 16, 8, 16, 5, 8, 1>,
    &cutlass_gemm_scatter<128, 256, 64, 64, 64, 64, 16, 8, 16, 3, 1, 1>,
    &cutlass_gemm_scatter<128, 256, 64, 64, 64, 64, 16, 8, 16, 3, 2, 1>,
    &cutlass_gemm_scatter<128, 256, 64, 64, 64, 64, 16, 8, 16, 3, 3, 1>,
    &cutlass_gemm_scatter<128, 256, 64, 64, 64, 64, 16, 8, 16, 3, 4, 1>,
    &cutlass_gemm_scatter<128, 256, 64, 64, 64, 64, 16, 8, 16, 3, 6, 1>,
    &cutlass_gemm_scatter<128, 256, 64, 64, 64, 64, 16, 8, 16, 3, 8, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 3, 1, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 3, 2, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 3, 3, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 3, 4, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 3, 6, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 3, 8, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 4, 1, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 4, 2, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 4, 3, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 4, 4, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 4, 6, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 4, 8, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 5, 1, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 5, 2, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 5, 3, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 5, 4, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 5, 6, 1>,
    &cutlass_gemm_scatter<256, 128, 32, 64, 64, 32, 16, 8, 16, 5, 8, 1>,
    &cutlass_gemm_scatter<256, 128, 64, 64, 64, 64, 16, 8, 16, 3, 1, 1>,
    &cutlass_gemm_scatter<256, 128, 64, 64, 64, 64, 16, 8, 16, 3, 2, 1>,
    &cutlass_gemm_scatter<256, 128, 64, 64, 64, 64, 16, 8, 16, 3, 3, 1>,
    &cutlass_gemm_scatter<256, 128, 64, 64, 64, 64, 16, 8, 16, 3, 4, 1>,
    &cutlass_gemm_scatter<256, 128, 64, 64, 64, 64, 16, 8, 16, 3, 6, 1>,
    &cutlass_gemm_scatter<256, 128, 64, 64, 64, 64, 16, 8, 16, 3, 8, 1>,
};
