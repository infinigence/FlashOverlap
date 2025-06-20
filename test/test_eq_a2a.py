import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os
import json

from torch.profiler import profile, record_function, ProfilerActivity

torch.ops.load_library("../build/lib/libst_pybinding.so")

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

class SPModel(nn.Module):
    def __init__(self, rank, world_size, dim, M, BM, BN, hint, mlen, cseg, algo, nccl_id):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        # Linear层输入输出维度为2048 (32x64)
        self.linear = nn.Linear(dim, dim, dtype=torch.float16).to(rank)

        torch.cuda.set_device(rank)

        self.overlap_class = torch.classes.flashoverlap_class.OverlapImpl()
        self.overlap_class.nccl_init(rank, world_size, nccl_id)
        self.overlap_class.cutlass_init()
        self.overlap_class.overlap_init()

        tm, tn = div_up(M, BM), div_up(dim, BN)
        self.algo = algo
        self.counter = torch.zeros((tm + 1, tn), dtype=torch.int).to(rank)
        self.reorder_array = reorder_indices(tm * tn, hint).reshape((tm, tn))

        self.cseg_cpu = torch.tensor(cseg, dtype=torch.int32) 
        self.cseg_gpu = self.cseg_cpu.cuda(rank)

        self.mlen_cpu = torch.tensor(mlen, dtype=torch.int32)

    def forward(self, x):
        """ 
        输入形状: [2, 44, 80, 32, 64] (32x64=2048)
        输出形状: [16, 44, 80, 4, 64] (4x64=256)
        """
        # 1. 保存原始形状
        original_shape = x.shape  # [2,44,80,32,64]

        x = x.reshape(original_shape[0] * original_shape[1] * original_shape[2], 
            original_shape[3] * original_shape[4])

        # 创建输出缓冲区 [64,44*80*64]
        y = torch.zeros(
            x.shape[0], x.shape[1], 
            device=x.device,
            dtype=x.dtype
        )
        output_tensor = torch.zeros(
            x.shape[0], x.shape[1], 
            device=x.device,
            dtype=x.dtype
        )

        self.overlap_class.gemm_eq_all2all_overlap(x, self.linear.weight, y, output_tensor, 
            self.counter, self.reorder_array, 1, self.cseg_cpu, self.cseg_gpu, self.mlen_cpu, self.algo, False)

        output_tensor = output_tensor.reshape(
            original_shape[0] * self.world_size,
            original_shape[1], 
            original_shape[2], 
            original_shape[3] // self.world_size, 
            original_shape[4]
        )

        return output_tensor

class SPBaselineModel(nn.Module):
    def __init__(self, rank, world_size, dim):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        # Linear层输入输出维度为2048 (32x64)
        self.linear = nn.Linear(dim, dim, dtype=torch.float16).to(rank)

    def forward(self, x):
        """ 
        输入形状: [2, 44, 80, 32, 64] (32x64=2048)
        输出形状: [16, 44, 80, 4, 64] (4x64=256)
        """
        # 1. 保存原始形状
        original_shape = x.shape  # [2,44,80,32,64]

        # 2. 执行Linear计算（最后两维展平为2048）
        x = x.reshape(*original_shape[:-2], -1)  # [2,44,80,2048]
        x = self.linear(x)  # 输出 [2,44,80,2048]
        # 恢复为 [2,44,80,8,4,64]
        x = x.reshape(original_shape[0], original_shape[1],
            original_shape[2], self.world_size,
            original_shape[3] // self.world_size, original_shape[4])  

        # 3. 准备All-to-All通信
        # [8,2,44,80,4,64]
        x = x.permute(3, 0, 1, 2, 4, 5).contiguous()

        # 展平非通信维度
        x = x.reshape(x.shape[0], -1)  # [8,2*44*80*4*64]

        # 4. 配置通信参数
        send_chunks = 1
        recv_chunks = send_chunks 

        # 创建输出缓冲区 [8,2*44*80*4*64]
        output_tensor = torch.zeros(
            x.shape[0], x.shape[1], 
            device=x.device,
            dtype=x.dtype
        )

        # 5. 执行all_to_all_single
        dist.all_to_all_single(
            output_tensor,
            x,
            output_split_sizes=[recv_chunks]*self.world_size, 
            input_split_sizes=[send_chunks]*self.world_size
        )

        # 6. 恢复最终形状 [16, 44, 80, 4, 64]
        output_tensor = output_tensor.reshape(
            original_shape[0] * self.world_size, original_shape[1], original_shape[2], 
            original_shape[3] // self.world_size, original_shape[4]
        ).contiguous()

        return output_tensor

def generate_mapping_sequence(data, G, rank):
    # 筛选满足 (index % (G * 2)) // 2 == rank 的 index
    filtered = [x for x in data if (x % (G * 2)) // 2 == rank]
    if not filtered:
        return []

    # 生成目标顺序的序列：G * 2 * k + 2*rank, G * 2 * k + 2*rank +1
    max_x = max(filtered)
    target_sequence = []
    k = 0
    while True:
        x1 = G * 2 * k + 2 * rank
        x2 = x1 + 1
        if x1 > max_x:
            break
        if x1 in filtered:
            target_sequence.append(x1)
        if x2 <= max_x and x2 in filtered:
            target_sequence.append(x2)
        k += 1

    # 生成 filtered 中 x 到其索引的映射
    value_to_pos_in_filtered = {x: i for i, x in enumerate(filtered)}

    # 获取 target_sequence 中 x 在 filtered 中的原始索引
    mapping = [value_to_pos_in_filtered[x] for x in target_sequence if x in value_to_pos_in_filtered]

    assert len(mapping) == len(data) // G, f"Expected mapping size {len(data) // G}, got {len(mapping)}"

    local_len = len(mapping)
    result = []
    for g in range(G):
        shifted = [x + g * local_len for x in mapping]
        result.extend(shifted)

    assert len(result) == len(set(result)), "Duplicate elements found in the result"

    return result

def test(rank, world_size, nccl_id, result_dict_0, result_dict_1):
    setup(rank, world_size)

    local_F = 2
    H = 44
    W = 80
    N = 32
    D = 64

    config_file = f'../configs/m7040n2048k2048_gefo.json'
    with open(config_file, 'r') as file:
        config = json.load(file)

    M = local_F * H * W
    BM = config["BM"][0]
    BN = config["BN"][0]
    hint = config["hint"]
    cSeg = config["cSeg"]
    Algo = config["Algo"][0]
    mLen = config["mLen"]

    # 模拟输入数据 [2,44,80,32,64] (32x64=2048)
    input_tensor = torch.randn(local_F, H, W, N, D, dtype=torch.float16).to(rank)

    # 创建模型
    model_ref = SPBaselineModel(rank, world_size, N * D).to(rank)
    output_ref = model_ref(input_tensor)

    model = SPModel(rank, world_size, N * D, M, BM, BN, hint, mLen, cSeg, Algo, nccl_id).to(rank)
    model.linear.weight = model_ref.linear.weight
    output = model(input_tensor)

    print(f"Rank {rank}: Input shape {input_tensor.shape}, Output shape {output.shape}")

    mapping = generate_mapping_sequence(hint, world_size, rank)
    mapping_gpu = torch.tensor(mapping, dtype=torch.int32, device=output.device)

    reordered_output = torch.empty_like(output)
    torch.ops.flashoverlap_op.reorder(output.reshape((-1, N // world_size * D)), reordered_output, BM, BN, 1, mapping_gpu)

    # [16, 44, 80, 4, 64]
    all_close = torch.allclose(reordered_output, output_ref, rtol=5e-2, atol=5e-2)
    print(f"Rank {rank}: Outputs are close: {all_close}")

    for _ in range(10):
        _ = model_ref(input_tensor)
    torch.cuda.synchronize()

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(100)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(100)]
    for i in range(100):
        start_event[i].record()
        _ = model_ref(input_tensor)
        end_event[i].record()
    torch.cuda.synchronize()
    dur_ref = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    for _ in range(10):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(100)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(100)]
    for i in range(100):
        start_event[i].record()
        _ = model(input_tensor)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    result_dict_0[rank] = torch.mean(dur_ref).item()
    result_dict_1[rank] = torch.mean(dur).item()

    cleanup()

if __name__ == "__main__":
    world_size = 8 # 16总batch/2 per GPU = 8 GPUs

    manager = mp.Manager()
    result_dict_0 = manager.dict()
    result_dict_1 = manager.dict()

    nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
    torch.cuda.synchronize()

    mp.spawn(test, args=(world_size, nccl_id, result_dict_0, result_dict_1), nprocs=world_size, join=True)

    dur_ref = torch.empty((world_size))
    dur = torch.empty((world_size))
    for i in range(world_size):
        dur_ref[i] = result_dict_0[i]
        dur[i] = result_dict_1[i]

    print("Baseline duration: {%.2f} ms, overlap duration: {%.2f}" % (dur_ref.max().item(), dur.max().item()))