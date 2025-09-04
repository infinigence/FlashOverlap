import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm

torch.ops.load_library("../build/lib/libst_pybinding.so")

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((dim), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def reorder_indices(S, hint):
    original = list(range(S))
    new_order = [-1] * S
    for i, element in enumerate(hint):
        new_order[element] = i
    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i
    return torch.tensor(new_order, dtype=torch.int, device="cuda")

def generate_random_block_reorder(M, N, BM, BN, device='cuda'):
    num_blocks_m = (M + BM - 1) // BM
    num_blocks_n = (N + BN - 1) // BN
    total_blocks = num_blocks_m * num_blocks_n
    reorder_indices = torch.randperm(total_blocks, dtype=torch.int, device=device)
    return reorder_indices

def perf_overlap(M, N, BM, BN):
    WARM_UP = 100
    REP = 1000
    rLDN = N // BN
    rmsnorm_layer = RMSNorm(N)
    C2 = torch.empty((M, N), dtype=torch.float16, device="cuda")
    D2 = torch.empty((M, N), dtype=torch.float16, device="cuda")
    ReorderedArray = generate_random_block_reorder(M, N, BM, BN).reshape(((M+BM-1)//BM, (N+BN-1)//BN))
    torch.ops.flashoverlap_op.reorder_rmsnorm(C2, D2, rmsnorm_layer.weight, BM, BN, rLDN, ReorderedArray)
    for _ in range(WARM_UP):
        torch.ops.flashoverlap_op.reorder_rmsnorm(C2, D2, rmsnorm_layer.weight, BM, BN, rLDN, ReorderedArray)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    for i in range(REP):
        start_event[i].record()
        torch.ops.flashoverlap_op.reorder_rmsnorm(C2, D2, rmsnorm_layer.weight, BM, BN, rLDN, ReorderedArray)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    return torch.mean(dur).item()

def perf_baseline(M, N):
    WARM_UP = 100
    REP = 1000
    rmsnorm_layer = RMSNorm(N)
    C1 = torch.empty((M, N), dtype=torch.float16, device="cuda")
    D1 = torch.empty((M, N), dtype=torch.float16, device="cuda")
    for _ in range(WARM_UP):
        torch.ops.flashoverlap_op.rmsnorm(C1, D1, rmsnorm_layer.weight)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    for i in range(REP):
        start_event[i].record()
        torch.ops.flashoverlap_op.rmsnorm(C1, D1, rmsnorm_layer.weight)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    return torch.mean(dur).item()

def test_pattern(M, N, pattern_name):
    if pattern_name == "tile":
        BM, BN = 256, 128
    elif pattern_name == "subtile":
        BM, BN = 64, 128
    elif pattern_name == "subtoken":
        BM, BN = 1, 128
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    if M % BM != 0:
        print(f"Warning: M={M} is not divisible by BM={BM}")
    if N % BN != 0:
        print(f"Warning: N={N} is not divisible by BN={BN}")
    try:
        baseline_time = perf_baseline(M, N)
        overlap_time = perf_overlap(M, N, BM, BN)
        overhead = overlap_time / baseline_time
        return baseline_time, overlap_time, overhead
    except Exception as e:
        print(f"Error testing {pattern_name} pattern: {e}")
        return None, None, None

def main():
    M_list = [2 ** i for i in range(8, 15)]
    N_list = [2 ** i for i in range(10, 15)]
    
    patterns = ["tile", "subtile", "subtoken"]
    results = {pattern: {"overheads": []} for pattern in patterns}
    
    total_cases = len(M_list) * len(N_list)

    combinations = [(M, N, pattern) for M in M_list for N in N_list for pattern in patterns]

    for M, N, pattern in tqdm(combinations, desc="Testing cases"):
        baseline, overlap, overhead = test_pattern(M, N, pattern)
        if baseline is not None:
            results[pattern]["overheads"].append(overhead)
    
    print(f"Total cases tested: {total_cases}")
    print(f"M range: {min(M_list)} to {max(M_list)}")
    print(f"N range: {min(N_list)} to {max(N_list)}")
    
    print("\n=== Overhead Results ===")
    print(f"{'Pattern':<10} Average Overhead ± {'Std Dev (%)':<10}")
    print("-" * 40)

    for pattern in patterns:
        overheads = results[pattern]["overheads"]
        if overheads:
            overhead_mean = np.mean(overheads) * 100 - 100
            overhead_std = np.std(overheads) * 100
            print(f"{pattern:<10} {overhead_mean:.2f}% ± {overhead_std:.2f}%")
        else:
            print(f"{pattern:<10} {'Failed':<20} {'Failed':<10}")

if __name__ == "__main__":
    main()