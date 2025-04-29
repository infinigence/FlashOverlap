<div align="center">

# FlashOverlap 

<a href="https://arxiv.org/abs/2504.19519">
    <img src="https://img.shields.io/badge/FlashOverlap-Tech Report-red"></a>
<a href="https://zhuanlan.zhihu.com/p/1897633068380054002?share_code=1nCLEM5AgyjRb&utm_psn=1900536763014963236&utm_source=wechat_timeline&utm_medium=social&s_r=0">
    <img src="https://img.shields.io/badge/FlashOverlap-ZHIHU-blue"></a>

A Lightweight Design for Computation-Communication Overlap.
</div>

## Milestone
- [x] demo for GEMM+AllReduce
- [x] predictive search for wave grouping
- [ ] multi-node example
- [ ] demo for GEMM+ReduceScatter, GEMM+AlltoAll
- [ ] more platforms (e.g., hopper GPU)
- [ ] end2end example

## Build and Install
### Dependency
The main dependency is [NCCL](https://developer.nvidia.com/nccl/nccl-download), which *FlashOverlap* uses for communication. It is convenient to download from the official website. The code has been tested with `v2.18.3` and `v2.19.3`. 

Another dependency is [CUTLASS](https://github.com/NVIDIA/cutlass.git), which is included as submodule. Note than the code has been tested with `v3.6.0` and `v3.9.0`, but fails with `v3.4.0`. We assume `CUTLASS>=v3.6.0` works fine.  

The code onlu supports `sm_80, sm_86, sm_89` now, and the evaluation enviroments include NVIDIA RTX 3090, RTX 4090, A800, and A100 GPUs. The versions of CUDA Toolkit include `CUDA 12.1, 12.2`.

### Install
First, pull the repo:

```shell
    $ git clone https://github.com/infinigence/FlashOverlap.git
    $ git submodule update --init --recursive
```
Install PyTorch and other required packages through `pip` or `conda`:
```shell
    $ pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    $ pip install numpy==2.1.2, pandas==2.2.3, setuptools==75.8.0
```

Before compiling, generate the GEMM instances:
```shell
    $ mkdir ./configs
    $ cd ./tool
    $ python generate_instances.py
```

This repo uses cmake (>=3.18) for compiling:

```shell
    $ cmake -B build
    $ cmake --build build -j
```
Then the operators are registered as torch.class, and in Python code, the `.so` should be included whenever the operators are used.
```python
    torch.ops.load_library("../build/lib/libst_pybinding.so")
```

## Quick Start
### File Structure
```plaintext
.
├── cmake
│   └── Modules
│       └── FindNCCL.cmake
├── configs                   // To store GEMM and overlapping configs
├── example
│   ├── correctness.py        // Check correctness of GEMM+AllReduce+RMSNorm
├── src
│   ├── 3rdparty
│   ├── gemm                  // CUTLASS GEMM Wrappers
│   │   ├── gemm.cu
│   │   └── gemm.h
│   ├── inc                   // Instantiate templated GEMMs
│   ├── overlap               // Source files for signal+reorder
│   ├── rmsnorm               // Source files for reorder+RMSNorm
│   ├── tiling                // Tiling definition  
│   ├── baseline_impl.cu      // Baseline implementation class
│   ├── baseline_impl.h
│   ├── CMakeLists.txt
│   ├── nccl_utils.cu         // NCCL id generation function
│   ├── nccl_utils.h
│   ├── overlap_impl.cu       // Overlap implementation class
│   ├── overlap_impl.h
│   ├── pybind.cpp
│   └── wait.cuh              // Signal kernel
├── test
│   └── test.py
├── tool
│   └── generate_instances.py
├── tune
│   ├── bandwidth.py
│   ├── gen_config.py
│   ├── profile_config.py
│   └── search.py
└── CMakeLists.txt
```

### Generate GEMM configuration
Currently the repo supports two ways to generate the proper configs for GEMMs for better performance. Only one GPU is needed for this operation. 

0. Make sure the `./configs` dir is created. 
```shell
    $ cd tune
```

1. Using the [CUTLASS Profiler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/profiler.md). Follow the README and wirte the profiling results in `$CSV_PATH/*.csv`. Then, generate the `.json` file in configs. 
```shell
    $ python profile_config.py --m $M --n $N --k $K --path $CSV_PATH
```

2. Using the customized profiler for a specific shape. The profiling process finishes within minutes. 
```shell
    $ python profile_config.py --m $M --n $N --k $K
```

### Tune
Tune the wave group size. Note multiple GPUs are needed in this program and the enviroment variable `CUDA_VISIBLE_DEVICES` must be set, as we use the `spawn` method (torch.multiprocessing.spawn) and the rank and world size are explicitly determined. 

1. The repo provides both the exhaustive and predictive methods, and the latter is recommended when `MxN>4096x4096`. If the predictive method is chosen, please generate the bandwidth curve first. Given GPU and primitive, the bandwidth curve needs only one generation. 
```shell
    $ CUDA_VISIBLE_DEVICES=0,1 python bandwidth.py --comm_op all_reduce
```
2. Two search methods share the same script, `--predictive_search` should be specified if used.
```shell
    $ CUDA_VISIBLE_DEVICES=0,1 python search.py --m $M --n $N --k $K --comm_op all_reduce --predictive_search True
```
3. The generated solution is written into the same `.json` file. 

### Speed Test
Open the test dir and run the script.
```shell
    $ cd ./test
    $ CUDA_VISIBLE_DEVICES=0,1 python test.py --m $M --n $N --k $K
```

### Correctness Test
1. Open the example dir.
```
    $ cd ./example
```

2. Evaluate the correctness of GEMM+AllReduce+RMSNorm. The RMSNorm must be included as the tile order is corrected in the kernel. 
```shell
    $ CUDA_VISIBLE_DEVICES=0,1 python correctness.py --m $M --n $N --k $K
```
3. We define the `ReorderRMSNorm` class in `RMSNorm.py` and the `OverlapRowParallelLayer` class in `RowParallelLayer.py`, which can replace the `RMSNorm` class and `RowParallelLayer` class, respectively. It's a simple example of usage in end-to-end inference or training. 

## Citation
```
    @misc{hong2025flashoverlap,
      title={FlashOverlap: A Lightweight Design for Efficiently Overlapping Communication and Computation},
      author={Ke Hong, Xiuhong Li, Minxu Liu, Qiuli Mao, Tianqi Wu, Zixiao Huang, Lufang Chen, Zhong Wang, Yichong Zhang, Zhenhua Zhu, Guohao Dai, Yu Wang},
      year={2025},
      eprint={2504.19519},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
    }
```
