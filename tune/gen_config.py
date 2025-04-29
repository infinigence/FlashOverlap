'''
    Using a single GPU for the program:
        e.g., python3 gen_config.py --m 4096 --n 8192 --k 8192 --path $YOUR_DIR
'''

import torch
import argparse
import pandas as pd
import json
import os
from pathlib import Path

torch.ops.load_library("../build/lib/libst_pybinding.so")

def get_optimal(data, idx: int):
    data["Runtime"] = data["Runtime"].astype(float)
    data_sorted = data.sort_values(by="Runtime")
    # get the optimal performance
    optimal = data_sorted.iloc[idx]
    return (optimal["cta_n"], optimal["cta_m"], optimal["cta_k"], \
        optimal["cta_n"]//optimal["warps_n"], optimal["cta_m"]//optimal["warps_m"], optimal["cta_k"]//optimal["warps_k"], \
        optimal["inst_m"], optimal["inst_n"], optimal["inst_k"], \
        optimal["stages"], optimal["swizzle_size"], optimal["split_k_slices"], optimal["Runtime"])

def perf_wrapped_gemm(M: int, N: int, K: int, Algo: int):
    cutlass_gemm = torch.classes.flashoverlap_class.OverlapImpl()
    cutlass_gemm.cutlass_init()

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")

    for _ in range(10):
        cutlass_gemm.cutlass_gemm(A, B, C, Algo)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(100)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(100)]
    for i in range(100):
        start_event[i].record()
        cutlass_gemm.cutlass_gemm(A, B, C, Algo)
        end_event[i].record()
    torch.cuda.synchronize()
    cutlass_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    return torch.mean(cutlass_dur).item()

def read_algo_dict(file_path: str, key_tuple: tuple):
    if Path(file_path).exists():
        # 如果文件存在，加载字典
        # print(f"Config {key_tuple} found.")
        data_dict = torch.load(file_path, weights_only=True)
    else:
        # 如果文件不存在，创建一个新的空字典
        data_dict = {}
    
    if key_tuple in data_dict:
        return data_dict[key_tuple]
    else:
        # 如果key_tuple不在字典中，
        # 1. 告诉用户需要修改gemm_tiling.cuh，并重新编译
        print(f"Config {key_tuple} not found in the dictionary. Please update gemm_tiling.cuh and recompile.")
        # 2. 分配一个新的序号
        new_index = len(data_dict)
        data_dict[key_tuple] = new_index
        print(data_dict)
        # 3. 并重新保存dict
        torch.save(data_dict, file_path)
        return new_index

def save_json(M: int, N: int, K: int, bm_list, bn_list, idx_list, dur_list):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}.json'
    if Path(file_path).exists():
        # 如果文件存在，加载 JSON 数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 如果文件不存在，创建一个空的字典
        data = {}
    data["BM"] = bm_list
    data["BN"] = bn_list
    data["dur"] = dur_list
    data["Algo"] = idx_list
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


# Define the main function
def main():

    # pass the problem size M, N, K via parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--path', type=str, default="../../../cutlass/build/")
    args = parser.parse_args()

    # Load the .csv file
    # generate the file name by combining M, N, K
    file_name = os.path.join(args.path, f'm{args.m}n{args.n}k{args.k}.gemm.csv')
    # check if the file exists, if not, report the error
    try:
        with open(file_name, 'r') as file:
            print(f'File found.')
    except FileNotFoundError:
        print(f'File {file_name} not found. Please generate the profiling results first.')
        return

    data = pd.read_csv(file_name)
    
    idx_list = []
    dur_list = []
    bm_list = []
    bn_list = []
    for idx in range(10):
        result = get_optimal(data, idx)

        assert result[-2] == 1, "SplitK is not supported with the implementation based on CUTLASS."
        # print(result)

        # check the GEMM performance under our wrapper
        # find the algo number in the dict, if does not exit, then add a new term
        algo_idx = read_algo_dict("../configs/AlgoDict.pt", result[:-1])
        # benchmark the wrapped GEMM performance
        t = perf_wrapped_gemm(args.m, args.n, args.k, algo_idx)
        idx_list.append(algo_idx)
        dur_list.append(t)
        bm_list.append(int(result[0]))
        bn_list.append(int(result[1]))

    # save the config into a .json file
    save_json(args.m, args.n, args.k, bm_list, bn_list, idx_list, dur_list)
    print("GEMM configs saved.")

if __name__ == "__main__":
    main()