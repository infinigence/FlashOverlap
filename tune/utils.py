import torch
from typing import List, Tuple

def get_swizzled_tile_order_direct(swizzle_size: int, cta_m: int, cta_n: int, m: int, n: int) -> List[Tuple[int, int]]:
    grid_m = (m + cta_m - 1) // cta_m
    grid_n = (n + cta_n - 1) // cta_n
    
    def get_log_tile(n_dim, N_val):
        if N_val >= 8 and n_dim >= 6:
            return 3
        elif N_val >= 4 and n_dim >= 3:
            return 2
        elif N_val >= 2 and n_dim >= 2:
            return 1
        else:
            return 0
    
    log_tile = get_log_tile(grid_n, swizzle_size)
    tile_size = 1 << log_tile
    
    grid_x = grid_m * tile_size
    grid_y = (grid_n + tile_size - 1) // tile_size
    
    all_tiles = []
    
    for block_idx_y in range(grid_y):
        for block_idx_x in range(grid_x):
            if (grid_m < swizzle_size) or (grid_n < swizzle_size):
                ti = block_idx_x
                tj = block_idx_y
            else:
                if log_tile > 0:
                    ti = block_idx_x >> log_tile
                    offset = block_idx_x & (tile_size - 1)
                    tj = (block_idx_y << log_tile) + offset
                else:
                    ti = block_idx_x // swizzle_size
                    offset = block_idx_x % swizzle_size
                    tj = block_idx_y * swizzle_size + offset
            
            if ti < grid_m and tj < grid_n:
                all_tiles.append((ti, tj))
    
    return all_tiles

def get_grouped_tile_index(tile_order: List[Tuple[int, int]], group_size: int = 106) -> List[int]:
    total_tiles = len(tile_order)
    flat_indices = []

    original_order = sorted(tile_order, key=lambda x: (x[0], x[1]))
    tile_to_original_idx = {tile: idx for idx, tile in enumerate(original_order)}
    
    for group_start in range(0, total_tiles, group_size):
        group_end = min(group_start + group_size, total_tiles)
        group_tiles = tile_order[group_start:group_end]
        
        group_indices = [tile_to_original_idx[tile] for tile in group_tiles]
        flat_indices.extend(group_indices)
    
    return flat_indices

def read_algo_dict(algo_idx: int, file_path: str = "../configs/AlgoDict.pt") -> int:
    try:
        algo_dict = torch.load(file_path)
        
        if not isinstance(algo_dict, dict):
            raise ValueError(f"Loaded file {file_path} is not a dictionary")
        
        matching_keys = []
        for key, value in algo_dict.items():
            if value == algo_idx:
                matching_keys.append(key)
        
        if not matching_keys:
            available_values = list(set(algo_dict.values()))[:5]
            raise KeyError(f"Value {algo_idx} not found in algorithm dictionary. "
                          f"Available values (first 5): {available_values}")
        
        algo_key = matching_keys[0]
        if len(matching_keys) > 1:
            print(f"Warning: Multiple keys found for value {algo_idx}. Using first key: {algo_key}")
        
        if len(algo_key) < 2:
            raise IndexError(f"Key tuple {algo_key} is too short to get second last element")

        swizzle_size = algo_key[-2]

        return int(swizzle_size)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Algorithm dictionary file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading algorithm dictionary: {e}")