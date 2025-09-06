import numpy as np
from typing import List, Tuple, Union
import os
import argparse
import torch
import json

def get_swizzled_tile_order(swizzle_size: int, cta_m: int, cta_n: int, m: int, n: int) -> List[Tuple[int, int]]:
    grid_m = (m + cta_m - 1) // cta_m
    grid_n = (n + cta_n - 1) // cta_n
    
    all_tiles = [(ti, tj) for ti in range(grid_m) for tj in range(grid_n)]
    
    def calculate_log_tile(tiled_n, N):
        if N >= 8 and tiled_n >= 6:
            return 3
        elif N >= 4 and tiled_n >= 3:
            return 2
        elif N >= 2 and tiled_n >= 2:
            return 1
        else:
            return 0
    
    log_tile = calculate_log_tile(grid_n, swizzle_size)
    tile = 1 << log_tile
    
    tile_to_swizzled_idx = {}
    
    for ti, tj in all_tiles:
        linear_block_idx = ti * grid_n + tj
        
        grid_dim_x = grid_m * tile
        grid_dim_y = (grid_n + tile - 1) // tile
        
        block_idx_x = linear_block_idx % grid_dim_x
        block_idx_y = linear_block_idx // grid_dim_x
        
        if (grid_m < swizzle_size) or (grid_n < swizzle_size):
            swizzled_ti = block_idx_x
            swizzled_tj = block_idx_y
        else:
            if log_tile > 0:
                swizzled_ti = block_idx_x >> log_tile
                swizzled_tj = (block_idx_y << log_tile) + (block_idx_x & (tile - 1))
            else:
                swizzled_ti = block_idx_x // swizzle_size
                swizzled_tj = (block_idx_y * swizzle_size) + (block_idx_x % swizzle_size)
        
        swizzled_linear_idx = swizzled_ti * grid_n + swizzled_tj
        tile_to_swizzled_idx[(ti, tj)] = swizzled_linear_idx
    
    sorted_tiles = sorted(all_tiles, key=lambda tile: tile_to_swizzled_idx[tile])
    
    return sorted_tiles


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

def print_tile_order(tile_order: List[Tuple[int, int]], grid_m: int, grid_n: int):
    order_grid = np.full((grid_m, grid_n), -1, dtype=int)
    
    for execution_step, (ti, tj) in enumerate(tile_order):
        order_grid[ti, tj] = execution_step
        
    print("Tile Execution Order Grid:")
    print("(Number denotes tile execution order, starting from 0)")
    print(f"Grid Shape: ({grid_m}, {grid_n})")
    print(order_grid)

def find_matching_config(m: int, n: int, config_dir: str = "../configs"):
    pattern = f"m{m}n{n}k"
    
    print(f"Looking for pattern: {pattern}* in {config_dir}")
    
    matching_files = []
    for filename in os.listdir(config_dir):
        if filename.startswith(pattern) and (filename.endswith('.json') or filename.endswith('.pt')):
            matching_files.append(os.path.join(config_dir, filename))
    
    if matching_files:
        return matching_files[0]
    
    return None

def parse_algo_params_from_pt(algo_id: int, algo_dict_path: str = "../configs/AlgoDict.pt"):
    
    try:
        algo_dict = torch.load(algo_dict_path, map_location='cpu')
        print(algo_dict)
        
        matching_keys = []
        for key, value in algo_dict.items():
            if int(value) == algo_id:
                matching_keys.append(key)
        
        if matching_keys:
            key = matching_keys[0]
            
            cta_m = int(key[0])
            cta_n = int(key[1])
            swizzle_size = int(key[-2])
            return cta_m, cta_n, swizzle_size
        else:
            closest_key = None
            min_diff = float('inf')
            
            for key, value in algo_dict.items():
                if isinstance(value, (int, float)):
                    diff = abs(value - algo_id)
                    if diff < min_diff:
                        min_diff = diff
                        closest_key = key
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    diff = abs(value[0] - algo_id)
                    if diff < min_diff:
                        min_diff = diff
                        closest_key = key
            
            if closest_key:
                print(f"Warning: Algorithm ID {algo_id} not found, using closest ID {min_diff} away")
                params = algo_dict[closest_key]
                if isinstance(params, torch.Tensor):
                    params = params.tolist()
                cta_m = int(params[0])
                cta_n = int(params[1])
                swizzle_size = int(params[-2])
                return cta_m, cta_n, swizzle_size
            
            raise ValueError(f"Algorithm ID {algo_id} not found in AlgoDict.pt")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"AlgoDict.pt file not found at {algo_dict_path}")
    except Exception as e:
        raise ValueError(f"Error loading AlgoDict.pt: {e}")

def read_config_file(config_path: str):
    if config_path.endswith('.pt'):
        try:
            config_data = torch.load(config_path, map_location='cpu')
            if not isinstance(config_data, dict):
                raise ValueError("PyTorch config file should contain a dictionary")
            return config_data
        except Exception as e:
            raise ValueError(f"Error loading PyTorch config file: {e}")
    else:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(config_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    return json.loads(content)
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error with encoding {encoding}: {e}")
        
        raise ValueError("Could not decode config file with any supported encoding")
    
def print_grouped_tile_index(tile_order: List[Tuple[int, int]], group_size: int = 106):
    print(f"\nGrouped Tile Index (group size: {group_size}):")
    print("-" * 60)
    
    total_tiles = len(tile_order)
    num_groups = (total_tiles + group_size - 1) // group_size
    
    original_order = sorted(tile_order, key=lambda x: (x[0], x[1]))
    tile_to_original_idx = {tile: idx for idx, tile in enumerate(original_order)}
    
    execution_to_original = [tile_to_original_idx[tile] for tile in tile_order]
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, total_tiles)
        
        group_execution_indices = list(range(start_idx, end_idx))
        
        group_original_indices = execution_to_original[start_idx:end_idx]
        
        group_original_indices_sorted = sorted(group_original_indices)
        
        print(f"Group {group_idx} (Execution steps {start_idx}-{end_idx-1}):")
        print(f"  Original tile indices in execution order: {group_original_indices}")
        print(f"  Original tile indices sorted: {group_original_indices_sorted}")
        
        print(f"  Corresponding tiles in execution order: {tile_order[start_idx:end_idx]}")
        
        unique_rows = len(set(ti for ti, tj in tile_order[start_idx:end_idx]))
        unique_cols = len(set(tj for ti, tj in tile_order[start_idx:end_idx]))
        print(f"  Unique rows: {unique_rows}, Unique columns: {unique_cols}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Calculate tile execution order based on CUTLASS swizzle')
    parser.add_argument('--m', type=int, required=True, help='Matrix M dimension size')
    parser.add_argument('--n', type=int, required=True, help='Matrix N dimension size')
    parser.add_argument('--config-dir', type=str, default='../configs', help='Path to config directory')
    parser.add_argument('--algo-dict', type=str, default='../configs/AlgoDict.pt', help='Path to AlgoDict.pt file')
    args = parser.parse_args()
    
    m = args.m
    n = args.n
    config_dir = args.config_dir
    algo_dict_path = args.algo_dict
    
    print(f"Looking for config for M={m}, N={n}")
    
    if not os.path.exists(config_dir):
        print(f"Config directory {config_dir} does not exist")
        return
    
    config_file = find_matching_config(m, n, config_dir)
    if not config_file:
        print(f"No matching config file found for m{m}n{n} in {config_dir}")
        print("Available files in config directory:")
        for f in os.listdir(config_dir):
            if f.endswith(('.json', '.pt')):
                print(f"  {f}")
        return
    
    print(f"Found config file: {config_file}")
    
    try:
        config_data = read_config_file(config_file)
        
        if "Algo" not in config_data or not config_data["Algo"]:
            raise ValueError("No Algo found in config file")
        
        algo_id = config_data["Algo"][0]
        if not isinstance(algo_id, int):
            try:
                algo_id = int(algo_id)
                print(f"Converted algo ID to integer: {algo_id}")
            except (ValueError, TypeError):
                raise ValueError(f"Algorithm ID should be integer, got {type(algo_id)}: {algo_id}")
        
        print(f"Using algorithm ID: {algo_id}")
        
        if not os.path.exists(algo_dict_path):
            print(f"AlgoDict.pt file not found at {algo_dict_path}")
            return
        
        cta_m, cta_n, swizzle_size = parse_algo_params_from_pt(algo_id, algo_dict_path)
        
        print(f"Parameters from config: cta_m={cta_m}, cta_n={cta_n}, swizzle_size={swizzle_size}")
        print("-" * 50)
        
        tile_order = get_swizzled_tile_order_direct(2, cta_m, cta_n, m, n)
        
        grid_m = (m + cta_m - 1) // cta_m
        grid_n = (n + cta_n - 1) // cta_n
        
        print_tile_order(tile_order, grid_m, grid_n)

        # print_grouped_tile_index(tile_order, 106)
        
        print("\nDetailed execution sequence (first 20 steps):")
        for step, (ti, tj) in enumerate(tile_order[:20]):
            print(f"Step {step:2d}: Tile ({ti}, {tj})")
        if len(tile_order) > 20:
            print(f"... and {len(tile_order) - 20} more steps")
            
    except Exception as e:
        print(f"Error processing config file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()