import torch
import argparse
import pandas as pd
import json
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
import glob
import os
import re
from tqdm import tqdm

torch.ops.load_library("../build/lib/libst_pybinding.so")

def div_up(x: int, y: int):
    return (x + y - 1) // y

def load_json(M: int, N: int, K: int, comm_op: str, world_size: int):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}_{comm_op}_{world_size}.json'
    
    assert Path(file_path).exists(), "Please run preprocess.py first!"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["BM"], data["BN"], data["dur"], data["Algo"], data["hint"]

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
        
        mod_values = chunk_row_ids % world_size
        
        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]
        
        reordered_row_id[current_row : current_row + chunk_size] = reordered_chunk
        current_row += chunk_size
    
    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)
    
    return remap

def interpolate_latency(samples, x, comm_op, world_size):
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    data_sizes = samples[:, 0].numpy()
    bandwidths = samples[:, 1].numpy()
    x_np = x.numpy()

    y_np = np.interp(x_np, data_sizes, bandwidths)

    y = torch.tensor(y_np, dtype=torch.float32).item()

    if comm_op == "all_reduce":
        latency = x * 2 * 2 * (world_size - 1) / y / (1024 ** 3)
    elif comm_op == "reduce_scatter":
        latency = x * 2 * (world_size - 1) / y / (1024 ** 3)

    return latency.item()

def predict_lat(M: int, N: int, gemm_dur: float, 
    comm_array: torch.Tensor, gp: list, tile_num: int, comm_op: str, world_size: int):

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    
    acc_comm_dur = 0
    acc_comp_dur = 0
    iter_num = len(gp)

    if iter_num == 1:
        acc_comm_dur = interpolate_latency(comm_array, M*N // tile_num * gp[0], comm_op, world_size) + gemm_dur
        return acc_comm_dur

    old_wave_num = (tile_num + sm_count - 1) // sm_count
    new_wave_num = (tile_num + sm_count - 3) // (sm_count - 2)
    gemm_dur = gemm_dur / old_wave_num * new_wave_num

    for i in range(iter_num):
        if i == 0:
            comm_dur = 0
        else:
            comm_dur = interpolate_latency(comm_array, M*N // tile_num * gp[i - 1], comm_op, world_size)
        acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + comm_dur 
        acc_comp_dur += gemm_dur / new_wave_num * ((gp[i] + sm_count - 3) // (sm_count - 2))
    acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + interpolate_latency(comm_array, M*N // tile_num * gp[-1], comm_op, world_size)

    return acc_comm_dur

def reorder_indices(S, hint):
    original = list(range(S))
    new_order = [-1] * S

    for i, element in enumerate(hint):
        new_order[element] = i

    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i
    
    return torch.tensor(new_order, dtype=torch.int, device="cuda")

def perf_running_process(rank, world_size, nccl_id,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, cSeg: list, hint: list, 
    comm_op: str,
    result_dict):

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32) 
    cSeg_GPU = cSeg_CPU.cuda(rank)

    TileNum = div_up(M, BM) * div_up(N, BN) 

    torch.cuda.set_device(rank)

    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()

    gemm_class.nccl_init(rank, world_size, nccl_id)
    gemm_class.cutlass_init()
    gemm_class.overlap_init()

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")

    MonitoredMatrix = torch.zeros(((N+BN-1)//BN), dtype=torch.int, device="cuda")
    ReorderedArray = reorder_indices(TileNum, hint).reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)
    
    _warm_up = 20
    _freq = 200

    if len(cSeg) == 1:
        # No overlapping
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                gemm_class.gemm_allreduce(A, B, C, Algo)

            gemm_class.gemm_allreduce(A, B, C, Algo)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_allreduce(A, B, C, Algo)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                gemm_class.gemm_reducescatter(A, B, C, D, Algo)

            MonitoredMatrix[0] = 0
            gemm_class.gemm_reducescatter(A, B, C, D, Algo)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_reducescatter(A, B, C, D, Algo)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    else:
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        else:
            dur = torch.zeros((_freq))

    result_dict[rank] = torch.mean(dur).item()
    
def perf_running(M: int, N: int, K: int, 
    BM: int, BN: int, Algo: int, 
    cSeg: list, hint: list, comm_op: str, world_size: int):
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
            perf_running_process,
            args=(world_size, nccl_id, M, N, K, BM, BN, Algo, cSeg, hint, comm_op, result_dict),
            nprocs=world_size
        )

    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max()

def integer_partitions(n):
    result = []
    def helper(remaining, path):
        if remaining == 0:
            result.append(path)
            return
        for i in range(1, remaining + 1):
            helper(remaining - i, path + [i])
    helper(n, [])
    return result

def fast_search(M: int, N: int, K: int, comm_array: torch.Tensor, comm_op: str, world_size: int):
    # load the .json file
    BM, BN, gemm_dur, Algo, hint = load_json(M, N, K, comm_op, world_size)

    # get the SM count
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count

    tile_num = div_up(M, BM) * div_up(N, BN)
    wave_num = div_up(tile_num, (sm_count - 2))

    min_group_size = div_up(wave_num, 10)

    assert hint != None, "Tuning fails! Try to increase min_group_size manully."
    # print("Start predictive searching.")
    
    normalized_wave_num = div_up(wave_num, min_group_size)
    group_size_list = integer_partitions(normalized_wave_num)
    
    pred_error_list = []
    group_choice = len(group_size_list)
    est_dur_list = []
    act_dur_list = []
    for i in range(group_choice):
        gp = group_size_list[i]
        iter_num = len(gp)
        acc = 0
        # avoid cold start
        if iter_num > 5 and gp[0] > 2:
            continue
        for j in range(iter_num):
            if j < iter_num - 1:
                gp[j] = gp[j] * (sm_count - 2) * min_group_size
                acc += gp[j]
            else:
                gp[j] = min(gp[j] * (sm_count - 2) * min_group_size, tile_num - acc)
        est_dur = predict_lat(M, N, gemm_dur, comm_array, gp, tile_num, comm_op, world_size)
        act_dur = perf_running(M, N, K, BM, BN, Algo, gp, hint, comm_op, world_size)
        # print(f"Group size: {gp}, Estimated duration: {est_dur:.2f} ms, Actual duration: {act_dur:.2f} ms")

        pred_error = abs(est_dur - act_dur) / act_dur * 100
        pred_error_list.append(pred_error)

        est_dur_list.append(est_dur)
        act_dur_list.append(act_dur)
    
    min_est_index = est_dur_list.index(min(est_dur_list))
    corresponding_act = act_dur_list[min_est_index]
    actual_min_act = min(act_dur_list)

    relative_performance = actual_min_act / corresponding_act * 100

    return pred_error_list, relative_performance

def main():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    
    config_files = glob.glob(f'../configs/m*n*k*_{gpu_name}_*.json')

    import random
    if len(config_files) > 5:
        selected_files = random.sample(config_files, 5)
    else:
        selected_files = config_files

    all_errors = []
    all_rel_perfs = []

    for config_file in tqdm(selected_files, desc="Processing config files", unit="file"):
        try:
            filename = os.path.basename(config_file)
            pattern = r'm(\d+)n(\d+)k(\d+)_([a-z0-9]+)_([a-z_]+)_(\d+)\.json'
            match = re.search(pattern, filename)
            if match:
                M = int(match.group(1))
                N = int(match.group(2))
                K = int(match.group(3))
                comm_op = match.group(5)
                file_world_size = int(match.group(6))
                
                comm_array = torch.load(f"../configs/bandwidth_{comm_op}_gpu{file_world_size}.pt")
                
                error_list, rel_perf = fast_search(M, N, K, comm_array, comm_op, file_world_size)
                
                all_errors.extend(error_list)  
                all_rel_perfs.append(rel_perf)  

            else:
                pass

        except Exception as e:
            pass

    if all_errors:
        error_mean = np.mean(all_errors)
        error_std = np.std(all_errors)
        print(f"Error: {error_mean:.4f} ± {error_std:.4f}")
    else:
        print("Error: No valid error data collected.")

    if all_rel_perfs:
        perf_mean = np.mean(all_rel_perfs)
        perf_std = np.std(all_rel_perfs)
        print(f"Relative Perf of Predictive Search: {perf_mean:.2f}% ± {perf_std:.2f}%")
    else:
        print("Rel Perf: No valid performance data collected.")

if __name__ == "__main__":
    main()