'''
    Using multiprocessing for distributed running, 
    please specify the GPUs via CUDA_VISIBLE_DEVICES:
        e.g., CUDA_VISIBLE_DEVICES=0,1 python3 search.py --m 4096 --n 8192 --k 4096 --comm_op all_reduce --search_policy robust
'''

import torch
import argparse
import pandas as pd
import json
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
import time

torch.ops.load_library("../build/lib/libst_pybinding.so")

def div_up(x: int, y: int):
    return (x + y - 1) // y

def load_json(M: int, N: int, K: int):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}.json'
    
    assert Path(file_path).exists(), "Please run preprocess.py first!"
    
    # 如果文件存在，加载 JSON 数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["BM"], data["BN"], data["dur"], data["Algo"]

def save_solution(M: int, N: int, K: int, BM: int, BN: int, gemm_dur: float, Algo: int, hint: list, cSeg: list):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}.json'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data["hint"] = hint
    data["cSeg"] = cSeg
    data["rLDN"] = 1
    data["BM"] = BM
    data["BN"] = BN
    data["dur"] = gemm_dur
    data["Algo"] = Algo
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

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

def compute_hint_process(rank, world_size, nccl_id,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: list, wSize: int, comm_op: str, 
    result_dict):

    TileNum = div_up(M, BM) * div_up(N, BN)
    # TN=1 aliases the monitor global counter with a segment counter in the CUDA kernel.
    # Tile order is deterministic in this degenerate layout, so skip monitor mode.
    if div_up(N, BN) == 1:
        result_dict[rank] = (
            True, list(range(TileNum)), [TileNum], [list(range(TileNum))],
            [], None, [TileNum],
        )
        return
    WaveNum = div_up(TileNum, wSize)

    cSeg = []
    for i in range(WaveNum):
        this_seg = min(wSize, TileNum - i * wSize)
        cSeg = cSeg + [this_seg]

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32) 
    cSeg_GPU = cSeg_CPU.cuda(rank)

    torch.cuda.set_device(rank)

    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()

    gemm_class.nccl_init(rank, world_size, nccl_id)
    gemm_class.cutlass_init()
    gemm_class.overlap_init()

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")

    MonitoredMatrix = torch.zeros(((M+BM-1)//BM + 1, (N+BN-1)//BN), dtype=torch.int, device="cuda") # TODO: We should put it in class
    ReorderedArray = torch.arange(0, TileNum, dtype=torch.int, device="cuda").reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)
    
    _warm_up = 100
    _sample = 10
    _probe = 50

    if comm_op == "all_reduce":
        for _ in range(_warm_up):
            gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
        
        samples = torch.empty((_sample, TileNum), dtype=torch.int, device="cuda")
        for i in range(_sample):
            MonitoredMatrix[0] = 0
            gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
            samples[i, :] = MonitoredMatrix[1:, :].view(-1)
    
    elif comm_op == "reduce_scatter":
        for _ in range(_warm_up):
            gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
        
        samples = torch.empty((_sample, TileNum), dtype=torch.int, device="cuda")
        for i in range(_sample):
            MonitoredMatrix[0] = 0
            gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
            samples[i, :] = MonitoredMatrix[1:, :].view(-1)

    else:
        assert comm_op in ["all_reduce", "reduce_scatter"], \
            f"comm_op must be 'all_reduce' or 'reduce_scatter', but got '{comm_op}'"

    torch.cuda.synchronize(rank)
    profile_start = time.perf_counter()
    for _ in range(_probe):
        if comm_op == "all_reduce":
            gemm_class.gemm_allreduce_overlap(
                A, B, C, MonitoredMatrix, ReorderedArray, 1,
                cSeg_CPU, cSeg_GPU, Algo, False,
            )
        else:
            gemm_class.gemm_reducescatter_overlap(
                A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1,
                cSeg_CPU, cSeg_GPU, Algo, False,
            )
    torch.cuda.synchronize(rank)
    profile_latency = (time.perf_counter() - profile_start) * 1000 / _probe

    # Local conservative grouping: keep each stable tile in its observed wave.
    # If a tile jitters across several waves, assign it to the last wave in
    # that observed range, and append it after the stable tiles of that wave.
    # This preserves the candidate while delaying only the affected local
    # boundary tiles, rather than moving every unstable tile to the task tail.
    stable_by_wave = [[] for _ in range(WaveNum)]
    jitter_by_wave = [[] for _ in range(WaveNum)]
    sample_waves = torch.div(samples, wSize, rounding_mode='floor')
    for tile in range(TileNum):
        observed = torch.unique(sample_waves[:, tile]).tolist()
        observed = [min(max(int(w), 0), WaveNum - 1) for w in observed]
        last_wave = max(observed)
        if len(observed) == 1:
            stable_by_wave[last_wave].append(tile)
        else:
            jitter_by_wave[last_wave].append(tile)
    wave_groups = [stable_by_wave[w] + jitter_by_wave[w] for w in range(WaveNum)]
    groups = [g for g in wave_groups if g]
    hint = [tile for group in groups for tile in group]
    safe_cSeg = [len(group) for group in groups]
    result_dict[rank] = (
        True, hint, safe_cSeg, wave_groups, [], profile_latency, cSeg,
    )

def compute_hint(M: int, N: int, K: int,
    BM: int, BN: int, Algo: list, wSize: int, comm_op: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
    torch.cuda.synchronize()
    # print(f"NCCL ID generated: {nccl_id[0]}")

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
            compute_hint_process,
            args=(world_size, nccl_id, M, N, K, BM, BN, Algo, wSize, comm_op, result_dict),
            nprocs=world_size
        )

    result = result_dict[0]
    measured = [result_dict[r][5] for r in range(world_size)
                if result_dict[r][5] is not None]
    if measured:
        result = result[:5] + (max(measured), result[6])
    return result

def interpolate_latency(samples, x, comm_op):
    world_size = torch.cuda.device_count()
    # 确保输入是 PyTorch 张量
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # 将数据转换为 NumPy 数组
    data_sizes = samples[:, 0].numpy()  # 数据量
    bandwidths = samples[:, 1].numpy()  # 带宽
    x_np = x.numpy()  # 需要插值的数据量

    # 使用 NumPy 的 interp 函数进行线性插值
    y_np = np.interp(x_np, data_sizes, bandwidths)

    # 将结果转换回 PyTorch 张量
    y = torch.tensor(y_np, dtype=torch.float32).item()

    # 使用 torch.interp 进行线性插值
    if comm_op == "all_reduce":
        latency = x * 2 * 2 * (world_size - 1) / y / (1024 ** 3)
    elif comm_op == "reduce_scatter":
        latency = x * 2 * (world_size - 1) / y / (1024 ** 3)

    return latency.item()

def predict_lat(M: int, N: int, gemm_dur: float, 
    comm_array: torch.Tensor, gp: list, tile_num: int, comm_op: str):

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    
    acc_comm_dur = 0
    acc_comp_dur = 0
    iter_num = len(gp)

    if iter_num == 1:
        acc_comm_dur = interpolate_latency(comm_array, M*N // tile_num * gp[0], comm_op) + gemm_dur
        return acc_comm_dur

    old_wave_num = (tile_num + sm_count - 1) // sm_count
    new_wave_num = (tile_num + sm_count - 3) // (sm_count - 2)
    gemm_dur = gemm_dur / old_wave_num * new_wave_num

    for i in range(iter_num):
        if i == 0:
            comm_dur = 0
        else:
            comm_dur = interpolate_latency(comm_array, M*N // tile_num * gp[i - 1], comm_op)
        acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + comm_dur 
        acc_comp_dur += gemm_dur / new_wave_num * ((gp[i] + sm_count - 3) // (sm_count - 2))
    acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + interpolate_latency(comm_array, M*N // tile_num * gp[-1], comm_op)

    return acc_comm_dur

def predict_overlap_latency(M, N, gemm_dur, comm_array, cseg, tile_num,
                            comm_op, profile_latency=None, profile_cseg=None):
    """Predict overlap latency, calibrated by the existing hint profiling."""
    estimate = predict_lat(
        M, N, gemm_dur, comm_array, cseg, tile_num, comm_op,
    )
    if profile_latency is None or not profile_cseg:
        return estimate
    profile_estimate = predict_lat(
        M, N, gemm_dur, comm_array, profile_cseg, tile_num, comm_op,
    )
    if profile_estimate <= 0:
        return estimate
    return estimate * profile_latency / profile_estimate

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
    cSeg: list, hint: list, comm_op: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
    torch.cuda.synchronize()
    # print(f"NCCL ID generated: {nccl_id[0]}")

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

def exhaustive_search(M: int, N: int, K: int, comm_op: str):
    # load the .json file
    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)

    # get the SM count
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count

    hint = None
    for t in range(5):
        BM = BM_list[t]
        BN = BN_list[t]
        gemm_dur = gemm_dur_list[t]
        Algo = Algo_list[t]

        tile_num = div_up(M, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, (sm_count - 2))

        #compute hint
        result = compute_hint(M, N, K, BM, BN, Algo, (sm_count - 2), comm_op)

        if result[0] == True:
            hint = result[1]
            break

    assert hint != None, "Tuning fails! Try to increase min_group_size manully."
    print("Start exhaustive searching.")

    min_dur = 1e5

    group_size_list = integer_partitions(wave_num)
    group_choice = len(group_size_list)
    for i in range(group_choice):
        gp = group_size_list[i]
        iter_num = len(gp)
        acc = 0
        for j in range(iter_num):
            if j < iter_num - 1:
                gp[j] = gp[j] * (sm_count - 2)
                acc += gp[j]
            else:
                gp[j] = min(gp[j] * (sm_count - 2), tile_num - acc)
        dur = perf_running(M, N, K, BM, BN, Algo, gp, hint, comm_op)
        print(gp, "%.4f" % (dur))

        if dur < min_dur:
            min_dur = dur
            cSeg = gp
        
    print("Best solution: ", cSeg)
    save_solution(M, N, K, BM, BN, gemm_dur, Algo, hint, cSeg)
    print("Solution saved.")


def predict_target_cseg(M, N, gemm_dur, comm_array, tile_num, wave_num,
                        min_group_size, sm_count, comm_op):
    best_est = 1e5
    best_cseg = None
    normalized_wave_num = div_up(wave_num, min_group_size)
    for gp0 in integer_partitions(normalized_wave_num):
        gp = list(gp0)
        if len(gp) > 5 and gp[0] > 2:
            continue
        acc = 0
        for j in range(len(gp)):
            if j < len(gp) - 1:
                gp[j] = gp[j] * (sm_count - 2) * min_group_size
                acc += gp[j]
            else:
                gp[j] = min(
                    gp[j] * (sm_count - 2) * min_group_size,
                    tile_num - acc,
                )
        est = predict_lat(M, N, gemm_dur, comm_array, gp, tile_num, comm_op)
        if est < best_est:
            best_est = est
            best_cseg = gp
    assert best_cseg is not None
    return best_cseg


def merge_local_wave_groups(wave_groups, target_cseg, wSize, tile_num):
    merged_groups = []
    wave_pos = 0
    for target in target_cseg:
        wave_count = max(1, div_up(target, wSize))
        group = []
        for _ in range(wave_count):
            if wave_pos < len(wave_groups):
                group.extend(wave_groups[wave_pos])
            wave_pos += 1
        if group:
            merged_groups.append(group)

    hint = [tile for group in merged_groups for tile in group]
    safe_cseg = [len(group) for group in merged_groups]
    assert len(hint) == tile_num
    assert len(set(hint)) == tile_num
    assert sum(safe_cseg) == tile_num
    return hint, safe_cseg


def build_candidate(M, N, K, index, BM, BN, gemm_dur, Algo,
                    comm_array, comm_op, sm_count):
    tile_num = div_up(M, BM) * div_up(N, BN)
    wave_num = div_up(tile_num, sm_count - 2)
    min_group_size = div_up(wave_num, 10)
    wSize = min_group_size * (sm_count - 2)
    result = compute_hint(M, N, K, BM, BN, Algo, wSize, comm_op)
    target_cseg = predict_target_cseg(
        M, N, gemm_dur, comm_array, tile_num, wave_num,
        min_group_size, sm_count, comm_op,
    )
    hint, safe_cseg = merge_local_wave_groups(
        result[3], target_cseg, wSize, tile_num,
    )
    latency = float(predict_overlap_latency(
        M, N, gemm_dur, comm_array, safe_cseg, tile_num, comm_op,
        result[5], result[6],
    ))
    print(
        f'candidate={index} BM={BM} BN={BN} Algo={Algo} '
        f'target={target_cseg} safe={safe_cseg} '
        f'pred_latency={latency:.4f}'
    )
    return {
        'index': index, 'BM': BM, 'BN': BN,
        'gemm_dur': gemm_dur, 'Algo': Algo,
        'hint': hint, 'cSeg': safe_cseg, 'latency': latency,
    }


def fast_search(M: int, N: int, K: int, comm_array: torch.Tensor, comm_op: str):
    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    candidate_count = min(10, len(BM_list))
    best = None

    print(f'Start fast conservative search ({candidate_count} candidates max).')
    for t in range(candidate_count):
        try:
            best = build_candidate(
                M, N, K, t, BM_list[t], BN_list[t],
                gemm_dur_list[t], Algo_list[t],
                comm_array, comm_op, sm_count,
            )
            break
        except Exception as exc:
            print(f'candidate={t} Algo={Algo_list[t]} skipped: {exc}')

    assert best is not None, 'All conservative candidates failed.'
    measured_latency = float(perf_running(
        M, N, K, best['BM'], best['BN'], best['Algo'],
        best['cSeg'], best['hint'], comm_op,
    ))
    print(
        f"Fast candidate={best['index']} Algo={best['Algo']} "
        f"measured_latency={measured_latency:.4f} cSeg={best['cSeg']}"
    )
    save_solution(
        M, N, K, best['BM'], best['BN'], best['gemm_dur'],
        best['Algo'], best['hint'], best['cSeg'],
    )
    print('Solution saved.')


def robust_search(M: int, N: int, K: int, comm_array: torch.Tensor, comm_op: str):
    # Evaluate every top GEMM candidate after applying the local conservative
    # wave assignment. Choose by measured end-to-end overlap latency, not by
    # bare GEMM latency or by the first candidate that passes profiling.
    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    candidate_count = min(10, len(BM_list))
    candidates = []

    print(f'Start multi-candidate conservative search ({candidate_count} candidates).')
    for t in range(candidate_count):
        try:
            candidates.append(build_candidate(
                M, N, K, t, BM_list[t], BN_list[t],
                gemm_dur_list[t], Algo_list[t],
                comm_array, comm_op, sm_count,
            ))
        except Exception as exc:
            print(f'candidate={t} Algo={Algo_list[t]} skipped: {exc}')

    assert candidates, 'All conservative candidates failed.'
    finalists = sorted(candidates, key=lambda item: item['latency'])[:2]
    best = None
    for candidate in finalists:
        measured_latency = float(perf_running(
            M, N, K, candidate['BM'], candidate['BN'], candidate['Algo'],
            candidate['cSeg'], candidate['hint'], comm_op,
        ))
        print(
            f"Finalist candidate={candidate['index']} Algo={candidate['Algo']} "
            f"pred={candidate['latency']:.4f} measured={measured_latency:.4f}"
        )
        if best is None or measured_latency < best['measured_latency']:
            best = dict(candidate, measured_latency=measured_latency)

    print(
        f"Best candidate={best['index']} Algo={best['Algo']} "
        f"measured_latency={best['measured_latency']:.4f} cSeg={best['cSeg']}"
    )
    save_solution(
        M, N, K, best['BM'], best['BN'], best['gemm_dur'],
        best['Algo'], best['hint'], best['cSeg'],
    )
    print('Solution saved.')


# Define the main function
def main():
    world_size = torch.cuda.device_count()

    # pass the problem size M, N, K via parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--k', type=int, default=8192)
    parser.add_argument('--n', type=int, default=8192)
    parser.add_argument('--comm_op', type=str, default='all_reduce')
    parser.add_argument('--predictive_search', type=bool, default=False)
    parser.add_argument(
        '--search_policy',
        choices=('fast', 'robust'),
        default='fast',
        help='fast selects the first usable candidate; robust evaluates all candidates and benchmarks the predicted top two',
    )
    args = parser.parse_args()

    # Force to use predictive search if the workload is large
    if args.predictive_search or args.m * args.n > 33554432:
        comm_array = torch.load(f"../configs/bandwidth_{args.comm_op}_tp{world_size}.pt")
        print("Bandwidth curve captured.")
        search_fn = fast_search if args.search_policy == 'fast' else robust_search
        search_fn(args.m, args.n, args.k, comm_array, args.comm_op)
    else:
        # compute the optimal solution
        exhaustive_search(args.m, args.n, args.k, args.comm_op)

if __name__ == "__main__":
    main()