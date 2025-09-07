import os
import re
import subprocess
from pathlib import Path
from tqdm import tqdm

CONFIG_DIR = Path("../configs")
TEST_SCRIPT = Path("../test/test.py")
TEST_A2A_SCRIPT = Path("../test/test_a2a.py")

filename_pattern = re.compile(
    r'm(\d+)n(\d+)k(\d+)_([a-z0-9]+)_([a-z_]+)_(\d+)\.json'
)

def main():
    if not CONFIG_DIR.exists():
        print(f"Error: Config directory '{CONFIG_DIR}' does not exist.")
        return

    if not TEST_SCRIPT.exists():
        print(f"Error: Test script '{TEST_SCRIPT}' not found.")
        return

    if not TEST_A2A_SCRIPT.exists():
        print(f"Error: A2A test script '{TEST_A2A_SCRIPT}' not found.")
        return

    config_files = []
    for file_path in CONFIG_DIR.glob("*.json"):
        match = filename_pattern.match(file_path.name)
        if match:
            config_files.append((file_path, match))
        else:
            # print(f"⚠️ Skipping unmatched file: {file_path.name}")
            pass

    if not config_files:
        print("❌ No valid config files found.")
        return

    for file_path, match in tqdm(config_files, desc="Running tests", unit="test"):
        m = int(match.group(1))
        n = int(match.group(2))
        k = int(match.group(3))
        comm_op = match.group(5)
        world_size = int(match.group(6))

        try:
            if comm_op in ["all_reduce", "reduce_scatter"]:
                cmd = [
                    "python", str(TEST_SCRIPT),
                    '--m', str(m),
                    '--n', str(n),
                    '--k', str(k),
                    '--world_size', str(world_size),
                    '--comm_op', str(comm_op)
                ]
                tqdm.write(f"🚀 Running: python test.py {m} {n} {k} {comm_op} {world_size}")
                subprocess.run(cmd, check=True, capture_output=True, text=True)

            elif comm_op == "all_to_all":
                cmd = [
                    "python", str(TEST_A2A_SCRIPT),
                    '--m', str(m),
                    '--n', str(n),
                    '--k', str(k),
                    '--world_size', str(world_size)
                ]
                tqdm.write(f"🚀 Running: python test_a2a.py {m} {n} {k} {world_size}")
                subprocess.run(cmd, check=True, capture_output=True, text=True)

            else:
                tqdm.write(f"🟡 Skipping unsupported comm_op: {comm_op}")
                continue

        except subprocess.CalledProcessError as e:
            tqdm.write(f"❌ Failed: {' '.join(cmd)}")
            tqdm.write(f"Error output:\n{e.stderr}")

    tqdm.write("✅ All tests completed.")

    import pandas as pd
    import numpy as np

    df = pd.read_csv('results.csv')

    df['decomposition_speedup'] = df['baseline'] / df['decomposition']
    df['flashoverlap_speedup'] = df['baseline'] / df['flashoverlap']

    grouped = df.groupby(['primitive', 'gpu'])

    results = []
    for (primitive, gpu), group in grouped:
        # Baseline is always 1 (reference point)
        baseline_str = "1.00(1.00-1.00)"
        
        # Decomposition speedup statistics
        dec_mean = group['decomposition_speedup'].mean()
        dec_min = group['decomposition_speedup'].min()
        dec_max = group['decomposition_speedup'].max()
        
        # FlashOverlap speedup statistics
        flo_mean = group['flashoverlap_speedup'].mean()
        flo_min = group['flashoverlap_speedup'].min()
        flo_max = group['flashoverlap_speedup'].max()
        
        results.append({
            'primitive': primitive,
            'gpu': gpu,
            'baseline': baseline_str,
            'decomposition': f"{dec_mean:.2f}({dec_min:.2f}-{dec_max:.2f})",
            'flashoverlap': f"{flo_mean:.2f}({flo_min:.2f}-{flo_max:.2f})"
        })

    result_df = pd.DataFrame(results)

    print("Speedup Statistics by Primitive and GPU Combination:")
    print("=" * 95)
    print(f"{'Primitive':<15} {'GPU':<10} {'Baseline':<20} {'Decomposition':<20} {'FlashOverlap':<20}")
    print("-" * 95)

    for _, row in result_df.iterrows():
        print(f"{row['primitive']:<15} {row['gpu']:<10} {row['baseline']:<20} {row['decomposition']:<20} {row['flashoverlap']:<20}")

    print("=" * 95)
    print("Note: Speedup values shown as mean(min-max), where baseline = 1.00")

if __name__ == "__main__":
    main()