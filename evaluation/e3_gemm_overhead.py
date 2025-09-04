import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import json
import argparse
import glob
import numpy as np
import re

torch.ops.load_library("../build/lib/libst_pybinding.so")

WARM_UP = 100
REP = 1000

def read_config(data):
    return data if isinstance(data, int) else data[0]

def generate_random_block_reorder(M, N, BM, BN, device='cuda'):
    num_blocks_m = (M + BM - 1) // BM
    num_blocks_n = (N + BN - 1) // BN
    total_blocks = num_blocks_m * num_blocks_n
    reorder_indices = torch.randperm(total_blocks, dtype=torch.int, device=device)
    return reorder_indices

def perf_gemm(M, N, K, config):
    Algo = read_config(config["Algo"])
    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()
    gemm_class.cutlass_init()
    gemm_class.overlap_init()
    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    for _ in range(WARM_UP):
        gemm_class.cutlass_gemm(A, B, C, Algo)
    torch.cuda.synchronize()
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    for i in range(REP):
        start_event[i].record()
        gemm_class.cutlass_gemm(A, B, C, Algo)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    return torch.mean(dur).item()

def perf_gemm_reorder_tile(M, N, K, config):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    wSize = sm_count - 2
    BM = read_config(config["BM"])
    BN = read_config(config["BN"])
    TileNum = (M + BM - 1) // BM * (N + BN - 1) // BN
    WaveNum = (TileNum + wSize - 1) // wSize
    rLDN = 1
    cSeg = []
    for i in range(WaveNum):
        this_seg = min(wSize, TileNum - i * wSize)
        cSeg.append(this_seg)
    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.cuda()
    Algo = read_config(config["Algo"])
    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()
    gemm_class.cutlass_init()
    gemm_class.overlap_init()
    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    MonitoredMatrix = torch.zeros(((M + BM - 1) // BM + 1, (N + BN - 1) // BN), dtype=torch.int, device="cuda")
    ReorderedArray = generate_random_block_reorder(M, N, BM, BN).reshape(((M + BM - 1) // BM, (N + BN - 1) // BN))
    for _ in range(WARM_UP):
        gemm_class.gemm_reorder_tile(A, B, C, MonitoredMatrix, ReorderedArray, rLDN, cSeg_CPU, cSeg_GPU, Algo)
    torch.cuda.synchronize()
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    for i in range(REP):
        start_event[i].record()
        gemm_class.gemm_reorder_tile(A, B, C, MonitoredMatrix, ReorderedArray, rLDN, cSeg_CPU, cSeg_GPU, Algo)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    return torch.mean(dur).item()

def perf_gemm_reorder_token(M, N, K, config):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    wSize = sm_count - 2
    BM = read_config(config["BM"])
    BN = read_config(config["BN"])
    TileNum = (M + BM - 1) // BM * (N + BN - 1) // BN
    WaveNum = (TileNum + wSize - 1) // wSize
    rLDN = 1
    cSeg = []
    for i in range(WaveNum):
        this_seg = min(wSize, TileNum - i * wSize)
        cSeg.append(this_seg)
    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.cuda()
    Algo = read_config(config["Algo"])
    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()
    gemm_class.cutlass_init()
    gemm_class.overlap_init()
    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    MonitoredMatrix = torch.zeros(((M + BM - 1) // BM + 1, (N + BN - 1) // BN), dtype=torch.int, device="cuda")
    ReorderedArray = generate_random_block_reorder(M, N, BM, BN).reshape(((M + BM - 1) // BM, (N + BN - 1) // BN))
    RowArray = torch.randperm(M * N // BN, dtype=torch.int, device="cuda").reshape((M * N // BN))
    for _ in range(WARM_UP):
        gemm_class.gemm_reorder_token(A, B, C, MonitoredMatrix, ReorderedArray, RowArray, rLDN, cSeg_CPU, cSeg_GPU, Algo)
    torch.cuda.synchronize()
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    for i in range(REP):
        start_event[i].record()
        gemm_class.gemm_reorder_token(A, B, C, MonitoredMatrix, ReorderedArray, RowArray, rLDN, cSeg_CPU, cSeg_GPU, Algo)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    return torch.mean(dur).item()

def main():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    
    config_files = glob.glob(f'../configs/m*n*k*_{gpu_name}.json')
    
    tile_overheads = []
    token_overheads = []
    first_config = True
    
    for config_file in config_files:
        try:
            filename = os.path.basename(config_file)
            pattern = r'm(\d+)n(\d+)k(\d+)'
            match = re.search(pattern, filename)
            if match:
                M = int(match.group(1))
                N = int(match.group(2))
                K = int(match.group(3))
                if gpu_name == 'a800':
                    K = int(match.group(3)) * 4
            else:
                print(f"Can read M, N, K from {filename}")
                continue
            
            with open(config_file, 'r') as file:
                config = json.load(file)
            
            if first_config:
                print("Running warmup for the first configuration...")
                perf_gemm(M, N, K, config)
                perf_gemm_reorder_tile(M, N, K, config)
                perf_gemm_reorder_token(M, N, K, config)
                print("Warmup completed. Now testing the first configuration again for statistics...")
                first_config = False
            
            gemm_time = perf_gemm(M, N, K, config)
            gemm_reorder_tile_time = perf_gemm_reorder_tile(M, N, K, config)
            tile_overhead = (gemm_reorder_tile_time - gemm_time) / gemm_time * 100
            tile_overheads.append(tile_overhead)

            gemm_reorder_token_time = perf_gemm_reorder_token(M, N, K, config)
            token_overhead = (gemm_reorder_token_time - gemm_time) / gemm_time * 100
            token_overheads.append(token_overhead)
            
        except Exception as e:
            print(f"Error processing {config_file}: {e}")
            continue
    
    if tile_overheads and token_overheads:
        tile_overheads_array = np.array(tile_overheads)
        token_overheads_array = np.array(token_overheads)
        
        tile_mean = np.mean(tile_overheads_array)
        tile_std = np.std(tile_overheads_array)
        token_mean = np.mean(token_overheads_array)
        token_std = np.std(token_overheads_array)
        
        print(f"\n=== Summary ===")
        print(f"Number of configurations tested: {len(tile_overheads)}")
        print(f"Tile reorder overhead: {tile_mean:.2f}% ± {tile_std:.2f}%")
        print(f"Token reorder overhead: {token_mean:.2f}% ± {token_std:.2f}%")
    else:
        print("No valid configurations were tested.")

if __name__ == "__main__":
    main()