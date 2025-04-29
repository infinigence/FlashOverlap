import torch
import argparse
import pandas as pd
from pathlib import Path
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

torch.ops.load_library("../build/lib/libst_pybinding.so")

# Function to initialize NCCL in each process
def perf_comm_process(rank, world_size, nccl_id, M, N, comm_op, result_dict):
    torch.cuda.set_device(rank)

    comm_class = torch.classes.flashoverlap_class.OverlapImpl()

    comm_class.nccl_init(rank, world_size, nccl_id)
    comm_class.cutlass_init()

    C = torch.empty((M, N), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

    if comm_op == "all_reduce":
        for _ in range(20):
            comm_class.nccl_allreduce(C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(200)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(200)]
        for i in range(200):
            start_event[i].record()
            comm_class.nccl_allreduce(C)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    elif comm_op == "reduce_scatter":
        for _ in range(20):
            comm_class.nccl_reducescatter(C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(200)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(200)]
        for i in range(200):
            start_event[i].record()
            comm_class.nccl_reducescatter(C)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    else:
        dur = 0

    result_dict[rank] = torch.mean(dur).item()

def perf_comm(M: int, N: int, comm_op: str):
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
            args=(world_size, nccl_id, M, N, comm_op, result_dict),
            nprocs=world_size
        )

    return result_dict[0]

# Define the main function
def main():
    world_size = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--comm_op', type=str, default="all_reduce")
    args = parser.parse_args()

    data_sizes = [i for i in range(2**20, 2**27, 2**20)] 
    bandwidths = []
    size_len = len(data_sizes)
    comm_array = torch.zeros((size_len, 2))
    for i, size in enumerate(data_sizes):
        # 创建输入张量（torch.float16）
        input_data = torch.randn(size, dtype=torch.float16, device='cuda')

        avg_time = perf_comm(128, size // 128, args.comm_op)
        
        # 计算带宽（单位：GB/s）
        data_size_bytes = input_data.numel() * input_data.element_size()
        if args.comm_op == "all_reduce":
            total_data_transferred = data_size_bytes * 2 * (world_size - 1) # AllReduce 的数据传输量
        elif args.comm_op == "reduce_scatter":
            total_data_transferred = data_size_bytes * (world_size - 1) 
        else:
            raise ValueError("Unsupported communication operation")

        bandwidth = (total_data_transferred / avg_time) / (1024 ** 3)  # 转换为 GB/s
        bandwidths.append(bandwidth)

        comm_array[i, 0] = size
        comm_array[i, 1] = bandwidth
        
    plt.plot(data_sizes, bandwidths, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('Data Size (elements)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Bandwidth vs Data Size')
    plt.grid(True)
    plt.savefig('bandwidth.png', dpi=300, bbox_inches='tight')
    plt.show()

    torch.save(comm_array, "../configs/bandwidth_" + args.comm_op + "_tp" + str(world_size) + ".pt")

if __name__ == "__main__":
    main()