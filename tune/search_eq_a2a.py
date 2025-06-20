import torch
import argparse
import pandas as pd
import json
from pathlib import Path
import torch.multiprocessing as mp

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

    return data["BM"][0], data["BN"][0], data["dur"][0], data["Algo"][0]

def save_solution(M: int, N: int, K: int, hint: list, cSeg: list, mlen: list):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}.json'

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data["hint"] = hint
    data["cSeg"] = cSeg
    data["rLDN"] = 1
    data["mLen"] = mlen
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def compute_hint_process(rank, world_size, nccl_id,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, wSize: int,
    result_dict):

    TileNum = div_up(M, BM) * div_up(N, BN)
    WaveNum = div_up(TileNum, wSize) 

    org_hint = [i for i in range(TileNum)]

    cSeg = []
    for i in range(WaveNum):
        this_seg = min(wSize, TileNum - i * wSize)
        cSeg = cSeg + [this_seg]

    mLen = count_mod_distribution_tensor(org_hint, 16, cSeg)
    # print(mLen)

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
    for w in range(WaveNum):
        index = torch.where(((samples >= w * wSize) * (samples < (w + 1) * wSize)).sum(dim=0) >= 9)

        if w < WaveNum - 1:
            assert index[0].shape[0] == wSize, "Inconsistency in test!"

        hint = hint + index[0].tolist()

    result_dict[rank] = hint

def compute_hint(M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, wSize: int):
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

    assert result_dict[0] == result_dict[1], "Inconsistency in test!"

    return result_dict[0]

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
    comm_op: str, mLen, 
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
    D = torch.empty((M, N), dtype=torch.float16, device="cuda")

    MonitoredMatrix = torch.zeros(((M+BM-1)//BM + 1, (N+BN-1)//BN), dtype=torch.int, device="cuda") # TODO: We should put it in class
    ReorderedArray = reorder_indices(TileNum, hint).reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    _warm_up = 20
    _freq = 200
    _using_streamk = False

    if len(cSeg) == 1:
        # No overlapping
        dur = torch.ones((_freq)) * 1e5
    else:
        if comm_op == 'all_to_all':
            for _ in range(_warm_up):
                gemm_class.gemm_eq_all2all_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, mLen, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_eq_all2all_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, mLen, Algo, False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        else:
            dur = torch.zeros((_freq))

    result_dict[rank] = torch.mean(dur).item()

def perf_running(M: int, N: int, K: int, 
    BM: int, BN: int, Algo: int, 
    cSeg: list, hint: list, comm_op: str, mlen):
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
            args=(world_size, nccl_id, M, N, K, BM, BN, Algo, cSeg, hint, comm_op, mlen, result_dict),
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

def sort_indexes_by_mod(indexes, M):
    # 创建一个字典来记录每个元素在原列表中的位置（保持原始顺序）
    position = {value: idx for idx, value in enumerate(indexes)}

    # 排序的关键是 (index % M, 原位置)，这样能保证先按组排，同组按原顺序
    sorted_indexes = sorted(indexes, key=lambda x: ((x % M) // 2, position[x]))

    return sorted_indexes

def count_mod_distribution_tensor(indexes, M, order_list):
    result = []
    start = 0
    for length in order_list:
        end = start + length
        sublist = indexes[start:end]
        mod_counts = [0] * (M // 2)
        for index in sublist:
            mod = index % M // 2
            mod_counts[mod] += 1
        result.append(mod_counts)
        start = end
    return torch.tensor(result, dtype=torch.int32)

def optimize_exhaustive(M: int, N: int, K: int, comm_op: str):
    # load the .json file
    BM, BN, gemm_dur, Algo = load_json(M, N, K)
    tile_num = div_up(M, BM) * div_up(N, BN)

    # get the SM count
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count

    # compute wave number
    # sm_count - 2 as overlap behavior occupies 2 SMs
    wave_num = div_up(tile_num, (sm_count - 2))

    #compute hint
    org_hint = compute_hint(M, N, K, BM, BN, Algo, (sm_count - 2))
    print("Start exhaustive searching.")

    # print(org_hint)
    hint = sort_indexes_by_mod(org_hint, 16)
    print(hint)

    min_dur = 1e5
    group_size_list = integer_partitions(wave_num)
    group_choice = len(group_size_list)
    for i in range(group_choice):
        gp = group_size_list[i]
        iter_num = len(gp)
        # figure out the mLen_CPU array
        acc = 0
        # if iter_num > 4 and gp[0] > 2:
        #     continue
        # avoid long tail
        # if iter_num > 4 and gp[-1] > 4: 
        #     continue
        if iter_num > 2:
            continue
        for j in range(iter_num):
            if j < iter_num - 1:
                gp[j] = gp[j] * (sm_count - 2)
                acc += gp[j]
            else:
                gp[j] = min(gp[j] * (sm_count - 2), tile_num - acc)
        mlen = count_mod_distribution_tensor(org_hint, 16, gp)
        print(mlen)
        dur = perf_running(M, N, K, BM, BN, Algo, gp, hint, comm_op, mlen)
        print(gp, "%.4f" % (dur))

        if dur < min_dur:
            min_dur = dur
            cSeg = gp
            mLen = mlen

    # if cSeg[-1] <= 2:
    #     cSeg[-2] += cSeg[-1]
    #     cSeg.pop()
    print("Best solution: ", cSeg)
    save_solution(M, N, K, hint, cSeg, mLen.tolist())
    print("Solution saved.")


# Define the main function
def main():

    # pass the problem size M, N, K via parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=512)
    parser.add_argument('--k', type=int, default=1792)
    parser.add_argument('--n', type=int, default=7168)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--comm_op', type=str, default='all_to_all')
    args = parser.parse_args()

    # compute the optimal solution
    optimize_exhaustive(args.m, args.n, args.k, args.comm_op)

if __name__ == "__main__":
    main()