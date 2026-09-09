<div align="center">

<img src="./docs/_static/image/FlashOverlap_LOGO.png" width="75" height="50">

# ***FlashOverlap*** 

<a href="https://arxiv.org/abs/2504.19519">
    <img src="https://img.shields.io/badge/FlashOverlap-Tech Report-red"></a>
<a href="https://zhuanlan.zhihu.com/p/1897633068380054002?share_code=1nCLEM5AgyjRb&utm_psn=1900536763014963236&utm_source=wechat_timeline&utm_medium=social&s_r=0">
    <img src="https://img.shields.io/badge/FlashOverlap-ZHIHU-blue"></a>

😊 **A Lightweight Design for Computation-Communication Overlap**
</div>

## News
**[2026.01.20]** Add multi-node scripts. 

**[2025.08.23]** *FlashOverlap* has been accepted by EuroSys'26 🎉 Tech report has been updated. 

## Roadmap
- [x] demo for GEMM+AllReduce
- [x] predictive search for wave grouping
- [x] multi-node example
- [x] demo for GEMM+ReduceScatter
- [ ] demo for GEMM+AlltoAll
- [x] code branch for AE
- [ ] more platforms (e.g., hopper GPU)
- [ ] end2end example

## How *FlashOverlap* Works
![FlashOverlap](./docs/_static/image/typical_timeline.jpeg)
The figure shows a typical timeline of computation-communication overlap in FlashOverlap. Two CUDA streams are for computation and communication, respectively. The CUTLASS kernel sends signals during GEMM computation in one stream, while a counting kernel stalls NCCL communication until receiving a preset number of signals in the other stream.

## Build and Install
### Dependency
The main dependency is [NCCL](https://developer.nvidia.com/nccl/nccl-download), which *FlashOverlap* uses for communication. It is convenient to download from the official website. The code has been tested with `v2.18.3` and `v2.19.3`. 

Another dependency is [CUTLASS](https://github.com/NVIDIA/cutlass.git), which is included as submodule. Note that the code has been tested with `v3.6.0` and `v3.9.0`, but fails with `v3.4.0`. We assume `CUTLASS>=v3.6.0` works fine.  

The code only supports `sm_80, sm_86, sm_89` now, and the evaluation enviroments include NVIDIA RTX 3090, RTX 4090, A800, and A100 GPUs. The versions of CUDA Toolkit include `CUDA 12.1, 12.2`.

### Install
First, pull the repo:

```shell
    $ git clone https://github.com/infinigence/FlashOverlap.git
    $ cd FlashOverlap
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
⚠️ ***Notice:*** the boundary handling is not implemented, thus the repo only supports regular GEMM shapes now (`M, N % 128 == 0`). 
### File Structure
```plaintext
.
├── cmake
│   └── Modules
│       └── FindNCCL.cmake
├── configs                   // To store GEMM and overlapping configs
├── example
│   ├── correctness_ar.py        // Check correctness of GEMM+AllReduce+RMSNorm
│   ├── correctness_rs.py        // Check correctness of GEMM+ReduceScatter+RMSNorm
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
│   ├── benchmark_overlap_variants.py // Compare fast and robust search-policy tuning time and overlap runtime latency
│   └── test.py
├── tool
│   └── generate_instances.py // Generate templated GEMMs
├── tune
│   ├── bandwidth.py          // Bandwidth test for predictive wave-group search
│   ├── gen_config.py         // Generate GEMM configs based on CUTLASS profiler
│   ├── profile_config.py     // Customized profiler
│   └── search.py             // Select wave grouping (exhaustive/predictive) and GEMM candidate (fast/robust)
└── CMakeLists.txt
```

### Generate GEMM configuration
Currently the repo supports two ways to generate the proper configs for GEMMs for better performance. Only one GPU is needed for this operation. 

0. Make sure the `./configs` dir is created. 
```shell
    $ cd tune
```

1. Using the [CUTLASS Profiler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/profiler.md). Follow the README and write the profiling results in `$CSV_PATH/*.csv`. Then, generate the `.json` file in configs. 
```shell
    $ python gen_config.py --m $M --n $N --k $K --path $CSV_PATH
```

2. Using the customized profiler for a specific shape. The profiling process finishes within minutes. (This method has been evaluated only on A800/A100 GPUs.)
```shell
    $ python profile_config.py --m $M --n $N --k $K
```

### Tune
Tune the wave group size. Note multiple GPUs are needed in this program and the environment variable `CUDA_VISIBLE_DEVICES` must be set, as we use the `spawn` method (torch.multiprocessing.spawn) and the rank and world size are explicitly determined. 

1. The repo provides exhaustive and predictive methods for selecting the wave grouping (`cSeg`) of a fixed GEMM candidate. Exhaustive search benchmarks every wave-group partition, whereas predictive wave-group search uses the measured GEMM latency and communication bandwidth curve to estimate the latency of each partition. Predictive wave-group search is recommended when `MxN>4096x4096`. Generate the bandwidth curve first; for a given GPU and communication primitive, it needs to be generated only once.
```shell
    $ CUDA_VISIBLE_DEVICES=0,1 python bandwidth.py --comm_op all_reduce
```
2. `--predictive_search` and `--search_policy` control different decisions. `--predictive_search` enables the model-based wave-group selection described above. Within that path, the separate `--search_policy` option controls GEMM-candidate selection: `fast` stops after the first usable candidate, while `robust` ranks multiple candidates using calibrated estimates of their final overlap latency and then benchmarks the predicted top two.
```shell
    $ CUDA_VISIBLE_DEVICES=0,1 python search.py --m $M --n $N --k $K --comm_op {all_reduce, reduce_scatter} --predictive_search True --search_policy {fast,robust}
```
3. The generated solution is written into the corresponding `.json` file. 

### Speed Test
Open the test dir and run the script.
```shell
    $ cd ./test
    $ CUDA_VISIBLE_DEVICES=0,1 python test.py --m $M --n $N --k $K --comm_op {all_reduce, reduce_scatter}
```

### Correctness Test
1. Open the example dir.
```
    $ cd ./example
```

2. Evaluate the correctness of GEMM+AllReduce+RMSNorm. The RMSNorm must be included as the tile order is corrected in the kernel. 
```shell
    $ CUDA_VISIBLE_DEVICES=0,1 python correctness_{ar, rs}.py --m $M --n $N --k $K
```
3. We define the `ReorderRMSNorm` class in `RMSNorm.py` and the `OverlapRowParallelLayer` class in `RowParallelLayer.py`, which can replace the `RMSNorm` class and `RowParallelLayer` class, respectively. It's a simple example of usage in end-to-end inference or training. 

### Multi-node Usage
⚠️ ***Notice:*** The `./configs` dir should be in a public storage, shared by the multiple nodes. 
1. Generate bandwidth curve.
```shell
    $ cd ./tune
    $ torchrun --nnodes=$NODE_NUM --nproc_per_node=$RANKS_PER_NODE --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR bandwidth_multinode.py --comm_op {all_reduce, reduce_scatter}
```

2. Search for the optimal configuration.
```shell
    $ torchrun --nnodes=$NODE_NUM --nproc_per_node=$RANKS_PER_NODE --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR search_multinode.py --m_dim $M --n_dim $N --k_dim $K --comm_op {all_reduce, reduce_scatter}
```

3. Test the speed.
```shell
    $ cd ./test
    $ torchrun --nnodes=$NODE_NUM --nproc_per_node=$RANKS_PER_NODE --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR test_multinode.py --m_dim $M --n_dim $N --k_dim $K --comm_op {all_reduce, reduce_scatter}
```

## Citation
```
    @article{hong2025flashoverlap,
      title={Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering},
      author={Ke Hong, Xiuhong Li, Minxu Liu, Qiuli Mao, Tianqi Wu, Zixiao Huang, Lufang Chen, Zhong Wang, Yichong Zhang, Zhenhua Zhu, Guohao Dai, Yu Wang},
      journal={arXiv preprint arXiv:2504.19519},
      year={2025}
    }
```
