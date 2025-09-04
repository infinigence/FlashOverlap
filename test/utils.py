import torch
import json
from pathlib import Path

def div_up(a, b):
    return (a + b - 1) // b

def load_json(M: int, N: int, K: int, world_size: int):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    sm_count = props.multi_processor_count
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}.json'

    assert Path(file_path).exists(), f"Config file {file_path} does not exist."

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hint = data["hint"]
    BM = data["BM"]
    BN = data["BN"]
    tile_num = div_up(M, BM) * div_up(N, BN)
    wave_num = div_up(tile_num, (sm_count - 2))
    min_group_size = div_up(wave_num, 10)
    
    return data["BM"], data["BN"], data["dur"], data["Algo"], hint, min_group_size

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

