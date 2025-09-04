import os
import glob
import random
import subprocess
import torch
import re

def main():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    
    config_files = glob.glob(f'../configs/m*n*k*_{gpu_name}_*.json')
    
    if len(config_files) > 10:
        selected_files = random.sample(config_files, 10)
    else:
        selected_files = config_files

    print("Randomly select 10 cases for correctness testing...")
    
    for config_file in selected_files:
        try:
            filename = os.path.basename(config_file)
            pattern = r'm(\d+)n(\d+)k(\d+)_([a-z0-9]+)_([a-z_]+)_(\d+)\.json'
            match = re.search(pattern, filename)
            
            if match:
                M = int(match.group(1))
                N = int(match.group(2))
                K = int(match.group(3))
                comm_op = match.group(5)
                world_size = int(match.group(6))
                
                print(f"Processing: M={M}, N={N}, K={K}, comm_op={comm_op}, world_size={world_size}")
                
                if comm_op == 'all_reduce':
                    script_path = '../example/correctness_ar.py'
                elif comm_op == 'reduce_scatter':
                    script_path = '../example/correctness_rs.py'
                else:
                    print(f"Unknown comm_op: {comm_op}, skipping...")
                    continue
                
                cmd = [
                    'python', script_path,
                    '--m', str(M),
                    '--n', str(N),
                    '--k', str(K),
                    '--gpu', str(world_size)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Successfully executed {filename}")
                    print(result.stdout)
                else:
                    print(f"Error executing {filename}:")
                    print(result.stderr)
                    
            else:
                print(f"Can't parse filename: {filename}")
                
        except Exception as e:
            print(f"Error processing {config_file}: {e}")
            continue

if __name__ == "__main__":
    main()