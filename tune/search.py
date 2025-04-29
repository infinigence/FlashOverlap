'''
    Using multiprocessing for distributed running, 
    please specify the GPUs via CUDA_VISIBLE_DEVICES:
        e.g., CUDA_VISIBLE_DEVICES=0,1 python3 search.py --m 4096 --n 8192 --k 4096 --comm_op all_reduce
'''

import torch
import argparse
import pandas as pd
import json
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np

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


def compute_hint_process(rank, world_size, nccl_id,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: list, wSize: int,
    result_dict):

    TileNum = div_up(M, BM) * div_up(N, BN)
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

    _warm_up = 100
    _sample = 10

    for _ in range(_warm_up):
        gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
    
    samples = torch.empty((_sample, TileNum), dtype=torch.int, device="cuda")
    for i in range(_sample):
        MonitoredMatrix[0] = 0
        gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
        samples[i, :] = MonitoredMatrix[1:, :].view(-1)

    hint = []
    is_consistency = True
    for w in range(WaveNum):
        index = torch.where(((samples >= w * wSize) * (samples < (w + 1) * wSize)).sum(dim=0) == 10)

        if w < WaveNum - 1:
            if index[0].shape[0] < wSize:
                is_consistency = False
                break

        hint = hint + index[0].tolist()
        
    result_dict[rank] = (is_consistency, hint)

def compute_hint(M: int, N: int, K: int,
    BM: int, BN: int, Algo: list, wSize: int):
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
            args=(world_size, nccl_id, M, N, K, BM, BN, Algo, wSize, result_dict),
            nprocs=world_size
        )

    return result_dict[0]

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
    
    _warm_up = 20
    _freq = 200

    if len(cSeg) == 1:
        # No overlapping
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

    else:
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

def optimize_exhaustive(M: int, N: int, K: int, comm_op: str):
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
        result = compute_hint(M, N, K, BM, BN, Algo, (sm_count - 2))

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


def fast_search(M: int, N: int, K: int, comm_array: torch.Tensor, comm_op: str):
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
        result = compute_hint(M, N, K, BM, BN, Algo, (sm_count - 2))

        if result[0] == True:
            hint = result[1]
            break

    assert hint != None, "Tuning fails! Try to increase min_group_size manully."
    print("Start predictive searching.")
    
    min_group_size = div_up(wave_num, 10)
    min_dur = 1e5
    normalized_wave_num = div_up(wave_num, min_group_size)
    group_size_list = integer_partitions(normalized_wave_num)
    
    group_choice = len(group_size_list)
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
        est_dur = predict_lat(M, N, gemm_dur, comm_array, gp, tile_num, comm_op)
        
        if est_dur < min_dur:
            min_dur = est_dur
            cSeg = gp
    print("Search process finished.")

    searched_lat = perf_running(M, N, K, BM, BN, Algo, cSeg, hint, comm_op)
    print("Searched latency: %.4f" % searched_lat)
    print("Best solution: ", cSeg)
    save_solution(M, N, K, BM, BN, gemm_dur, Algo, hint, cSeg)
    print("Solution saved.")


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
    args = parser.parse_args()

    # Force to use predictive search if the workload is large
    if args.predictive_search or args.m * args.n > 33554432:
        comm_array = torch.load(f"../configs/bandwidth_{args.comm_op}_tp{world_size}.pt")
        print("Bandwidth curve captured.")
        fast_search(args.m, args.n, args.k, comm_array, args.comm_op)
    else:
        # compute the optimal solution
        optimize_exhaustive(args.m, args.n, args.k, args.comm_op)


if __name__ == "__main__":
    main()