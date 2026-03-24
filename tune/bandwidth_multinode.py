import torch
import argparse
import os
import time
import torch.distributed as dist
from pathlib import Path
import matplotlib.pyplot as plt

# Load custom ops library
torch.ops.load_library("../build/lib/libst_pybinding.so")

# Constants
WARM_UP = 20
REP = 200

def init_distributed():
    """Initialize distributed training using pre-generated NCCL ID"""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(local_rank)

    # Initialize process group with timeout
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.default_pg_timeout,  # Use default timeout
            device_id=torch.device(f'cuda:{local_rank}')
        )
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group: {e}")
        print(f"Rank {rank}: MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
        raise

    return rank, local_rank, world_size

def perf_comm_test(M: int, N: int, comm_op: str):
    """Run communication performance test"""
    # rank, local_rank, world_size, nccl_id = init_distributed()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    # print(rank, local_rank, world_size)

    # Load NCCL ID from file (wait if not exists)
    # nccl_id_path = "../configs/nccl_id.pt"
    # while not os.path.exists(nccl_id_path):
        # time.sleep(0.1)
    # nccl_id = torch.load(nccl_id_path, weights_only=True)
    if rank == 0:
        nccl_id = torch.ops.flashoverlap_op.generate_nccl_id()
        nccl_id_tsr = torch.tensor(nccl_id, device=device)
        dist.broadcast(nccl_id_tsr, src=0)                   
    else:
        nccl_id_tsr = torch.zeros(16, dtype=torch.int64, device=device)
        dist.broadcast(nccl_id_tsr, src=0)
        nccl_id = nccl_id_tsr.cpu().tolist()
    # print(nccl_id)

    comm_class = torch.classes.flashoverlap_class.OverlapImpl()
    dist.barrier()
    comm_class.nccl_init(rank, world_size, nccl_id)
    comm_class.cutlass_init()

    C = torch.empty((M, N), dtype=torch.float16, device=device).normal_(mean=0., std=0.5)

    # Warm up
    if comm_op == "all_reduce":
        comm_func = comm_class.nccl_allreduce
    elif comm_op == "reduce_scatter":
        comm_func = comm_class.nccl_reducescatter
    else:
        raise ValueError(f"Unsupported comm_op: {comm_op}")

    for _ in range(WARM_UP):
        comm_func(C)

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

    for i in range(REP):
        start_events[i].record()
        comm_func(C)
        end_events[i].record()

    torch.cuda.synchronize()
    durations = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    avg_time = durations.mean().item()

    # Sync results across all ranks
    all_times = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_times, torch.tensor([avg_time], device=device))

    return torch.tensor(all_times).max().item()  # Return rank 0's result

def main():
    try:
        # Initialize distributed
        rank, _, world_size = init_distributed()
        
        # Ensure all processes are synchronized after initialization
        dist.barrier()
        if rank == 0:
            print(f"Successfully initialized distributed training with {world_size} processes")

        parser = argparse.ArgumentParser()
        parser.add_argument('--comm_op', type=str, default="all_reduce", choices=["all_reduce", "reduce_scatter"])
        args = parser.parse_args()

        data_sizes = [(int(2**(20 + 0.25*i)) // 1024 * 1024) for i in range(36)] 
        bandwidths = []
        comm_array = torch.zeros((len(data_sizes), 2))

        for i, size in enumerate(data_sizes):
            if rank == 0:
                print(f"Testing size: {size/2**20:.1f}MB")

            avg_time_ms = perf_comm_test(1024, size // 1024, args.comm_op)
            avg_time_s = avg_time_ms / 1000.0  # elapsed_time() returns ms, convert to s

            if rank == 0:
                # Calculate busbw (GB/s)
                data_size_bytes = size * 2  # float16 = 2 bytes
                if args.comm_op == "all_reduce":
                    busbw = data_size_bytes * 2 * (world_size - 1) / world_size / avg_time_s / (1024**3)
                else:  # reduce_scatter
                    busbw = data_size_bytes * (world_size - 1) / world_size / avg_time_s / (1024**3)

                bandwidths.append(busbw)
                comm_array[i, 0] = size
                comm_array[i, 1] = busbw

            dist.barrier()  # Sync before next size

        if rank == 0:
            # Plot results
            plt.figure(figsize=(10, 6))
            plt.plot(data_sizes, bandwidths, 'o-')
            plt.xscale('log', base=2)
            plt.yscale('log')
            plt.xlabel('Data Size (elements)')
            plt.ylabel('BusBW (GB/s)')
            plt.title(f'{args.comm_op} Bus Bandwidth (World Size: {world_size})')
            plt.grid(True, which="both", ls="-")
            plt.savefig(f'bandwidth_{args.comm_op}_ws{world_size}.png', dpi=300, bbox_inches='tight')

            # Save data
            torch.save(comm_array, f"../configs/bandwidth_{args.comm_op}_ws{world_size}.pt")
    except Exception as e:
        print(f"Error in main: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
