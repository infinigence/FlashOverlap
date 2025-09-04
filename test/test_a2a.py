import torch
import json
from pathlib import Path
import torch.multiprocessing as mp
import pandas as pd
import argparse
import os
from utils import generate_unbalanced_transfer_matrix, collect_workload

torch.ops.load_library("../build/lib/libst_pybinding.so")

WARM_UP=20
REP=200

def div_up(x: int, y: int):
    return (x + y - 1) // y

def reorder_indices(S, hint):
    # Generate the original array of indices [0, 1, ..., S-1]
    original = list(range(S))
    
    # Create an empty list to store the new order of indices
    new_order = [-1] * S
    
    # Place the indices of the hint list in the first positions of the new order
    for i, element in enumerate(hint):
        new_order[element] = i
    
    # Place the remaining indices in the new order
    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i
    
    return torch.tensor(new_order, dtype=torch.int, device="cuda")

def generate_row_remap_array(
    M, N, BM, BN, S_list, world_size, device="cuda"
):
    total_tiles = (M * N) // (BM * BN)
    assert sum(S_list) == total_tiles, "sum(S_list) must equal total number of tiles"
    
    original_row_ids = torch.arange(M * N // BN, dtype=torch.int, device=device)
    reordered_row_id = torch.empty_like(original_row_ids)
    
    current_row = 0
    for S in S_list:
        chunk_size = S * BM
        chunk_row_ids = original_row_ids[current_row : current_row + chunk_size]
        
        # Compute row_id % world_size for the current chunk
        mod_values = chunk_row_ids % world_size
        
        # Sort the chunk based on mod_values (stable sort)
        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]
        
        reordered_row_id[current_row : current_row + chunk_size] = reordered_chunk
        current_row += chunk_size
    
    # Compute remap: remap[original_row_id] = new_row_id
    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)
    
    return remap

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
    _using_streamk = False

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
    cSeg_list: list, hint_list: list, world_size: int, transfer_matrix, local_transfer_matrix):
    
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
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

# Function to initialize NCCL in each process
def perf_comm_process(rank, world_size, nccl_id, M, N, comm_type, result_dict):
    torch.cuda.set_device(rank)

    comm_class = torch.classes.flashoverlap_class.OverlapImpl()

    comm_class.nccl_init(rank, world_size, nccl_id)
    comm_class.cutlass_init()

    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    if comm_type == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")

    if comm_type == "all_reduce":
        for _ in range(WARM_UP):
            comm_class.nccl_allreduce(C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            comm_class.nccl_allreduce(C)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    elif comm_type == "reduce_scatter":
        for _ in range(WARM_UP):
            comm_class.nccl_reducescatter(C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            comm_class.nccl_reducescatter(C)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    else:
        dur = torch.zeros((REP))
        
    result_dict[rank] = torch.mean(dur).item()

def perf_comm(M: int, N: int, comm_type: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required!")
    
    nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
    torch.cuda.synchronize()
    # print(f"NCCL ID generated: {nccl_id[0]}")

    manager = mp.Manager()
    result_dict = manager.dict()

    # get the all reduce time
    mp.spawn(
            perf_comm_process,
            args=(world_size, nccl_id, M, N, comm_type, result_dict),
            nprocs=world_size
        )

    return result_dict[0]

def perf_baseline_process(rank, world_size, nccl_id, M, N, K, TopK, transfer_matrix, mode, result_dict):
    torch.cuda.set_device(rank)

    # **** Init Baseline Class **** #
    if mode == 'sequential':
        gemm_comm = torch.classes.flashoverlap_class.BaselineImpl()
    elif mode == 'decomposition':
        gemm_comm = torch.classes.flashoverlap_class.DecompositionImpl()
    else: 
        pass
    gemm_comm.nccl_init(rank, world_size, nccl_id)
    gemm_comm.cublas_init()

    local_token = transfer_matrix[rank, :].sum().item()
    A = torch.empty((local_token, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((local_token, N), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    D = torch.empty((TopK * world_size * M * N), dtype=torch.float16, device="cuda")

    
    # **** cuBLAS + NCCL **** #
    for _ in range(WARM_UP):
        gemm_comm.gemm_all2all(A, B, C, D, transfer_matrix)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    for i in range(REP):
        start_event[i].record()
        # torch.cuda.cudart().cudaProfilerStart()
        gemm_comm.gemm_all2all(A, B, C, D, transfer_matrix)
        # torch.cuda.cudart().cudaProfilerStop()
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    result_dict[rank] = torch.mean(dur).item()

def perf_baseline(M: int, N: int, K: int, TopK: int, world_size: int, transfer_matrix, mode: str):
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")
    # Use the custom NCCL initialization wrapper to get a unique NCCL ID
    # nccl_id = NcclInit()
    nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    # Spawn processes
    mp.spawn(
            perf_baseline_process,
            args=(world_size, nccl_id, M, N, K, TopK, transfer_matrix, mode, result_dict),
            nprocs=world_size
        )
    
    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]
    
    return dur.max()

def main():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    sm_count = props.multi_processor_count
    wave_size = sm_count - 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--k', type=int, default=8192)
    parser.add_argument('--n', type=int, default=8192)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--comm_op', type=str, default='all_to_all')
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    comm_op = args.comm_op
    world_size = args.world_size

    assert comm_op == "all_to_all", "Only all-to-all is supported in this script."

    m, n, k = args.m, args.n, args.k

    file_path = f'../configs/m{m}n{n}k{k}_{gpu_name}_{comm_op}_{world_size}.json'

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cSeg = data["cSeg"]
    transfer_matrix = generate_unbalanced_transfer_matrix(world_size * m, args.topk, world_size, 512, seed=7, max_attempts=100, imbalance_factor=5)
    workload_list, baseBM, baseBN, hint_list, _ = collect_workload(m, n, k, transfer_matrix)
    local_transfer_matrix = torch.tensor(data["transMat"], dtype=torch.int32)

    bm_list = []
    bn_list = []
    algo_list = []
    max_wave_num = 0
    gemm_dur = 0
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

    comm_dur = perf_comm(m, n, comm_op)
    overlap_dur = perf_running(m, n, k, args.topk, 
                bm_list, bn_list, algo_list, cSeg, hint_list, world_size, transfer_matrix, local_transfer_matrix)
    baseline_dur = perf_baseline(m, n, k, args.topk, world_size, transfer_matrix, 'sequential')

    speedup = baseline_dur / overlap_dur

    print(f"""
        {'Item':<10} {'Value':>15}
        {'-----':<10} {'-----':>15}
        {'m':<10} {m:>15}
        {'n':<10} {n:>15}
        {'k':<10} {k:>15}
        {'tile_num':<10} {tile_num:>15}
        {'comm_dur (ms)':<10} {comm_dur:>15.4f}
        {'baseline_dur (ms)':<10} {baseline_dur:>15.4f}
        {'overlap_dur (ms)':<10} {overlap_dur:>15.4f}
        {'speedup':<10} {speedup:>15.4f}
        """)

if __name__ == "__main__":
    main()




