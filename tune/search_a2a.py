import torch
import argparse
import pandas as pd
import json
from pathlib import Path
import torch.multiprocessing as mp
from search import compute_hint
import os

torch.ops.load_library("../build/lib/libst_pybinding.so")

def div_up(x: int, y: int):
    return (x + y - 1) // y

def load_json(M: int, N: int, K: int, world_size: int):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    sm_count = props.multi_processor_count
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}.json'

    if Path(file_path).exists():
        pass
    else:
        os.system('python3 profile_config.py --m %d --n %d --k %d' % (M, N, K))
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "hint" in data.keys():
        hint = data["hint"]
        BM = data["BM"]
        BN = data["BN"]
        tile_num = div_up(M, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, (sm_count - 2))
        min_group_size = div_up(wave_num, 10)
    else:
        hint = None
        for t in range(10):
            BM = data["BM"][t]
            BN = data["BN"][t]
            Algo = data["Algo"][t]
            
            tile_num = div_up(M, BM) * div_up(N, BN)
            wave_num = div_up(tile_num, (sm_count - 2))

            min_group_size = div_up(wave_num, 10)

            #compute hint
            # print(M, N, K, BM, BN, Algo)
            result = compute_hint(M, N, K, BM, BN, Algo, min_group_size * (sm_count - 2), 'all_reduce', world_size)

            if result[0] == True and (BM * BN == 256 * 128):
                hint = result[1]
                break

        assert hint != None, "Tuning fails! Try to increase min_group_size manully."
        data["BM"] = BM
        data["BN"] = BN
        data["rLDN"] = 1
        data["Algo"] = Algo
        data["dur"] = data["dur"][t]
        data["hint"] = hint
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    
    return data["BM"], data["BN"], data["dur"], data["Algo"], hint, min_group_size

def save_solution(M: int, N: int, K: int, gp_list: list, comm_op: str, world_size: int, local_transfer_matrix):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}_{comm_op}_{world_size}.json'
    
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}
    
    data["cSeg"] = gp_list
    data["rLDN"] = 1
    data["transMat"] = local_transfer_matrix.tolist()
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Function to initialize NCCL in each process
def perf_comm_process(rank, world_size, nccl_id, M, N, TopK, transfer_matrix, result_dict):
    torch.cuda.set_device(rank)

    comm_class = torch.classes.flashoverlap_class.OverlapImpl()

    comm_class.nccl_init(rank, world_size, nccl_id)
    comm_class.cutlass_init()
    local_token = transfer_matrix[rank, :].sum().item()

    C = torch.empty((local_token, N), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    D = torch.empty((TopK * world_size * M * N), dtype=torch.float16, device="cuda")

    for _ in range(20):
        comm_class.nccl_all2all(C, D, transfer_matrix)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(200)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(200)]
    for i in range(200):
        start_event[i].record()
        comm_class.nccl_all2all(C, D, transfer_matrix)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    
    result_dict[rank] = torch.mean(dur).item()

def perf_all2all(M, N, TopK, world_size, transfer_matrix):

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
            args=(world_size, nccl_id, M, N, TopK, transfer_matrix, result_dict),
            nprocs=world_size
        )

    return result_dict[0]

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

# def integer_partitions(n):
#     result = []
#     def helper(remaining, path, start):
#         if remaining == 0:
#             result.append(path)
#             return
#         for i in range(start, remaining + 1):
#             helper(remaining - i, path + [i], i)
#     helper(n, [], 1)
#     return result

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

def generate_unbalanced_transfer_matrix(token_num, topk, world_size, BM, seed=31, max_attempts=100, imbalance_factor=0.5):

    assert token_num % BM == 0, "token_num must be a multiple of BM"
    G = token_num // BM 
    assert G % world_size == 0, "token_num//BM must be divisible by world_size"
    groups_per_rank = G // world_size

    for attempt in range(max_attempts):
        current_seed = seed + attempt
        
        torch.manual_seed(current_seed)
        transfer_matrix = torch.zeros((world_size, world_size), dtype=torch.int32)

        weights = torch.softmax(torch.rand(world_size) * imbalance_factor, dim=0)
        
        for group_id in range(G):
            target_rank = group_id % world_size
            expert_choices = torch.multinomial(weights, topk, replacement=True)
            expert_counts = torch.bincount(expert_choices, minlength=world_size)
            transfer_matrix[:, target_rank] += expert_counts * BM

        row_sums = transfer_matrix.sum(dim=1)
        if not (row_sums == 0).any():
            return transfer_matrix
    
    raise RuntimeError(f"Failed to generate valid transfer matrix after {max_attempts} attempts. "
                      "Please check your parameters (world_size, topk, etc.).")

def collect_workload(M: int, N: int, K: int, transfer_matrix):
    world_size = transfer_matrix.size(0)
    workload_list = []

    BM_base = 0
    BN_base = 0
    hint_list = []
    max_mgs = 0
    for i in range(world_size):
        local_token = transfer_matrix[i, :].sum().item()
        BM, BN, dur, Algo, hint, mgs = load_json(local_token, N, K, world_size)
        workload_list.append((BM, BN, dur, Algo))

        if BM > BM_base:
            BM_base = BM
        if BN > BN_base:
            BN_base = BN

        hint_list.append(hint)
        max_mgs = max(mgs, max_mgs)

    return workload_list, BM_base, BN_base, hint_list, max_mgs

def count_indices_in_bins(indices, bins):
    
    bins_tensor = torch.as_tensor(bins)
    indices_tensor = torch.as_tensor(indices)
    
    if len(bins_tensor) == 0:
        return torch.tensor([], dtype=torch.int64)
    
    prefix_sum = torch.cumsum(bins_tensor, dim=0)
    start_points = torch.cat([
        torch.zeros(1, dtype=prefix_sum.dtype, device=prefix_sum.device),
        prefix_sum[:-1]
    ])
    
    i = torch.searchsorted(start_points, indices_tensor, side='right') - 1
    
    clamped_i = i.clamp(min=0, max=len(bins_tensor)-1)
    end_points = start_points[clamped_i] + bins_tensor[clamped_i]
    valid_mask = (i >= 0) & (i < len(bins_tensor)) & (indices_tensor < end_points)
    
    counts = torch.bincount(i[valid_mask], minlength=len(bins_tensor))
    return counts

def optimize_exhaustive(M: int, N: int, K: int, TopK: int, transfer_matrix):
    # collect the workload list
    workload_list, baseBM, baseBN, hint_list, max_mgs = collect_workload(M, N, K, transfer_matrix)
    # print(workload_list, baseBM, baseBN)

    world_size = transfer_matrix.size(0)

    # get the SM count
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count

    print("Start exhaustive searching.")

    max_wave_num = 0
    scaling_ratio_list = []
    tile_num_list = []
    bm_list = []
    bn_list = []
    algo_list = []
    for i in range(world_size):
        local_token = transfer_matrix[i, :].sum().item()
        BM = workload_list[i][0]
        BN = workload_list[i][1]
        tile_num = div_up(local_token, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, (sm_count - 2))
        tile_scaling_ratio = 1
        unified_wave_num = div_up(wave_num, tile_scaling_ratio)
        if unified_wave_num > max_wave_num:
            max_wave_num = unified_wave_num
        scaling_ratio_list.append(tile_scaling_ratio)
        tile_num_list.append(tile_num)
        bm_list.append(BM)
        bn_list.append(BN)
        algo_list.append(workload_list[i][-1])
    
    # print("Max wave number: ", max_wave_num)
    min_group_size = max_mgs

    min_dur = 1e5
    normalized_wave_num = div_up(max_wave_num, min_group_size)
    group_size_list = integer_partitions(normalized_wave_num)
    
    group_choice = len(group_size_list)
    for i in range(group_choice):
        gp_list = []
        iter_num = len(group_size_list[i])
        iter_transfer_matrix = torch.zeros((iter_num, world_size, world_size), dtype=torch.int32)
        for k in range(world_size):
            row_to_tile = workload_list[k][0] 
            gp = group_size_list[i].copy()
            acc = 0
            for j in range(iter_num):
                gp[j] = min(gp[j] * (sm_count - 2) * min_group_size * scaling_ratio_list[k], max(tile_num_list[k] - acc, 0)) 
                acc += gp[j]

                if gp[j] > 0:
                    this_hint = hint_list[k][acc - gp[j] : acc].copy()
                    # iter_transfer_matrix[j, k, :] = count_indices_in_bins(this_hint,  workload_list[k][1] * transfer_matrix[k, :])
                    iter_transfer_matrix[j, k, :] = count_indices_in_bins(this_hint,  N * transfer_matrix[k, :] // (workload_list[k][0] * workload_list[k][1]))
                    # print(N * transfer_matrix[k, :] // (workload_list[k][0] * workload_list[k][1]))
                
                # if j < iter_num - 1:
                #     gp[j] = gp[j] * (sm_count - 2) * scaling_ratio_list[k]
                #     acc += gp[j]
                # else:
                #     gp[j] = min(gp[j] * (sm_count - 2) * scaling_ratio_list[k], max(tile_num_list[k] - acc, 0))
            gp_list.append(gp)
        dur = perf_running(M, N, K, TopK, bm_list, bn_list, algo_list, gp_list, hint_list, world_size, 
            transfer_matrix, iter_transfer_matrix)
        print(gp_list, "%.4f" % (dur))

        if dur < min_dur:
            min_dur = dur
            cSeg = gp_list
            transMat = iter_transfer_matrix
        
    print("Best solution: ", cSeg)
    save_solution(M, N, K, cSeg, "all_to_all", world_size, transMat)
    print("Solution saved.")

# Define the main function
def main():

    # pass the problem size M, N, K via parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=512)
    parser.add_argument('--k', type=int, default=1792)
    parser.add_argument('--n', type=int, default=7168)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--comm_op', type=str, default='all2all')
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    world_size = args.world_size

    assert args.comm_op in ['all2all'], "Invalid communication operation!"

    # All token number is world_size * M
    transfer_matrix = generate_unbalanced_transfer_matrix(world_size * args.m, args.topk, world_size, 512, seed=7, max_attempts=100, imbalance_factor=5)
    print("Transfer matrix: ")
    print(transfer_matrix)

    # comm_dur = perf_all2all(args.m, args.n, args.topk, world_size, transfer_matrix)
    # print("NCCL %s dur: %.4f ms" % (args.comm_op, comm_dur))

    # compute the optimal solution
    optimize_exhaustive(args.m, args.n, args.k, args.topk, transfer_matrix)


if __name__ == "__main__":
    main()