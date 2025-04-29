file(GLOB gemm_files ${PROJECT_SOURCE_DIR}/src/gemm/*.cu)
add_library(gemm STATIC ${gemm_files})
target_include_directories(gemm PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/cutlass/include)
target_include_directories(gemm PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/cutlass/tools/util/include)
set_property(TARGET gemm PROPERTY POSITION_INDEPENDENT_CODE ON)

file(GLOB overlap_files ${PROJECT_SOURCE_DIR}/src/overlap/*.cu)
add_library(overlap STATIC ${overlap_files})
target_include_directories(overlap PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/cutlass/include)
target_include_directories(overlap PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/cutlass/tools/util/include)
set_property(TARGET overlap PROPERTY POSITION_INDEPENDENT_CODE ON)

file(GLOB norm_files ${PROJECT_SOURCE_DIR}/src/rmsnorm/*.cu)
add_library(norm STATIC ${norm_files})
set_property(TARGET norm PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(baseline_impl baseline_impl.cu)
set_property(TARGET baseline_impl PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries (baseline_impl PUBLIC gemm overlap norm)

add_library(overlap_impl overlap_impl.cu)
set_property(TARGET overlap_impl PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries (overlap_impl PUBLIC gemm overlap norm)

add_library(nccl_tool nccl_utils.cu)
set_property(TARGET nccl_tool PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(st_pybinding SHARED pybind.cpp)
target_link_libraries(
    st_pybinding
    baseline_impl
    overlap_impl
    nccl_tool
)

set_target_properties(st_pybinding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)
