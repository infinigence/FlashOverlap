import torch
import json
from pathlib import Path
import torch.multiprocessing as mp
import pandas as pd
import os
import numpy as np
import argparse
from utils import div_up, reorder_indices, generate_unbalanced_transfer_matrix

torch.ops.load_library("../../build/lib/libst_pybinding.so")

WARM_UP=100
REP=1000

def load_json(M: int, N: int, K: int):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    sm_count = props.multi_processor_count
    file_path = f'../../configs/m{M}n{N}k{K}_{gpu_name}.json'
    
    assert Path(file_path).exists(), "Please run the tuning first!"
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert "hint" in data.keys(), "Please run the tuning first!"
    hint = data["hint"]
    BM = data["BM"]
    BN = data["BN"]
    tile_num = div_up(M, BM) * div_up(N, BN)
    wave_num = div_up(tile_num, (sm_count - 2))
    # General group size threshold strategy
    min_group_size = div_up(wave_num, 10)
    
    return data["BM"], data["BN"], data["dur"], data["Algo"], hint, min_group_size

def collect_workload(M: int, N: int, K: int, transfer_matrix):
    world_size = transfer_matrix.size(0)
    workload_list = []

    BM_base = 0
    BN_base = 0
    hint_list = []
    max_mgs = 0
    for i in range(world_size):
        local_token = transfer_matrix[i, :].sum().item()
        BM, BN, dur, Algo, hint, mgs = load_json(local_token, N, K)
        workload_list.append((BM, BN, dur, Algo))

        if BM > BM_base:
            BM_base = BM
        if BN > BN_base:
            BN_base = BN

        hint_list.append(hint)
        max_mgs = max(mgs, max_mgs)

    return workload_list, BM_base, BN_base, hint_list, max_mgs

def perf_running_process(rank, world_size, nccl_id,
    M: int, N: int, K: int, TopK: int, 
    BM_list: list, BN_list: list, Algo_list: list, cSeg_list: list, hint_list: list, 
    transfer_matrix, local_transfer_matrix,
    result_dict):
    torch.cuda.set_device(rank)

    BM = BM_list[rank]
    BN = BN_list[rank]
    Algo = Algo_list[rank]
    cSeg = cSeg_list[rank]
    hint = hint_list[rank]

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32) 
    cSeg_GPU = cSeg_CPU.cuda(rank)

    local_token = transfer_matrix[rank, :].sum().item()
    TileNum = div_up(local_token, BM) * div_up(N, BN) 

    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()

    gemm_class.nccl_init(rank, world_size, nccl_id)
    gemm_class.cutlass_init()
    gemm_class.overlap_init()

    A = torch.empty((local_token, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((local_token, N), dtype=torch.float16, device="cuda")
    D = torch.empty((TopK * world_size * M * N), dtype=torch.float16, device="cuda")

    MonitoredMatrix = torch.zeros(((local_token+BM-1)//BM + 1, (N+BN-1)//BN), dtype=torch.int, device="cuda") # TODO: We should put it in class
    ReorderedArray = reorder_indices(TileNum, hint).reshape(((local_token+BM-1)//BM, (N+BN-1)//BN))
    
    _warm_up = 100
    _freq = 1000

    if len(cSeg) == 1:
        # No overlapping
        for _ in range(_warm_up):
            gemm_class.gemm_all2all(A, B, C, D, Algo, transfer_matrix)

        gemm_class.gemm_all2all(A, B, C, D, Algo, transfer_matrix)

        start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
        for i in range(_freq):
            start_event[i].record()
            gemm_class.gemm_all2all(A, B, C, D, Algo, transfer_matrix)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    else:
        for _ in range(_warm_up):
            gemm_class.gemm_all2all_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, local_transfer_matrix, Algo)

        MonitoredMatrix[0] = 0
        gemm_class.gemm_all2all_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, local_transfer_matrix, Algo)

        start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
        for i in range(_freq):
            start_event[i].record()
            gemm_class.gemm_all2all_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, local_transfer_matrix, Algo)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    result_dict[rank] = torch.mean(dur).item()
    
def perf_running(M: int, N: int, K: int, TopK: int, 
    BM_list: list, BN_list: list, Algo_list: list, 
    cSeg_list: list, hint_list: list, transfer_matrix, local_transfer_matrix):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = torch.ops.nil_op.generate_nccl_id()
    torch.cuda.synchronize()
    # print(f"NCCL ID generated: {nccl_id[0]}")

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
            perf_running_process,
            args=(world_size, nccl_id, M, N, K, TopK, BM_list, BN_list, Algo_list, cSeg_list, hint_list, 
                transfer_matrix, local_transfer_matrix, result_dict),
            nprocs=world_size
        )

    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max()

# Define the main function
def main():
    world_size = torch.cuda.device_count()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    sm_count = props.multi_processor_count
    wave_size = sm_count - 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=512)
    parser.add_argument('--k', type=int, default=1792)
    parser.add_argument('--n', type=int, default=7168)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--comm_op', type=str, default='all2all')
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    m, n, k, topk = args.m, args.n, args.k, args.topk

    transfer_matrix = generate_unbalanced_transfer_matrix(world_size * m, topk, world_size, 512, seed=7, max_attempts=100, imbalance_factor=5)
    workload_list, baseBM, baseBN, hint_list, _ = collect_workload(m, n, k, transfer_matrix)

    file_path = f'../../configs/m{m}n{n}k{k}_{gpu_name}_all_to_all_{world_size}.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cSeg = data["cSeg"]
    local_transfer_matrix = torch.tensor(data["transMat"], dtype=torch.int32)

    gemm_dur_list = []
    bm_list = []
    bn_list = []
    algo_list = []
    max_wave_num = 0
    gemm_dur = 0
    reorder_array_list = []
    for i in range(world_size):
        local_token = transfer_matrix[i, :].sum().item()
        BM = workload_list[i][0]
        BN = workload_list[i][1]
        tile_num = div_up(local_token, BM) * div_up(n, BN)
        wave_num = div_up(tile_num, (sm_count - 2))
        tile_scaling_ratio = baseBM // BM * baseBN // BN
        unified_wave_num = div_up(wave_num, tile_scaling_ratio)
        if unified_wave_num > max_wave_num:
            max_wave_num = unified_wave_num
        bm_list.append(BM)
        bn_list.append(BN)
        algo_list.append(workload_list[i][3])
        if gemm_dur < workload_list[i][2]:
            gemm_dur = workload_list[i][2]
            unified_tile_num = tile_num
        reorder_array_list.append(reorder_indices(tile_num, hint_list[i]).reshape((div_up(local_token, BM), div_up(n, BN))))

    # derive local reorder array after a2a      
    overlap_dur = perf_running(m, n, k, topk, 
        bm_list, bn_list, algo_list, cSeg, hint_list, transfer_matrix, local_transfer_matrix)

if __name__ == "__main__":
    main()