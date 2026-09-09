#!/usr/bin/env python3
"""Compare fast and robust search-policy tuning time and overlap runtime latency across several shapes."""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TUNE = ROOT / "tune"
TEST = ROOT / "test"
CONFIGS = ROOT / "configs"
DEFAULT_SHAPES = [
    "128x13696x8192",
    "128x40832x8192",
    "256x20608x8192",
    "512x10368x8192",
    "2048x4096x8192",
    "4096x4096x8192",
    "4096x8192x8192",
    "128x41088x4096",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, metavar="MxNxK")
    parser.add_argument("--repeats", type=int, default=3, help="test.py runs per variant")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / time.strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--comm-op", default="all_reduce")
    return parser.parse_args()


def parse_shape(text):
    values = tuple(int(value) for value in text.lower().split("x"))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"invalid shape {text!r}; expected MxNxK")
    return values


def run(command, cwd, env, log_path):
    start = time.perf_counter()
    process = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    log_path.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr)
    if process.returncode:
        raise RuntimeError(f"failed: {' '.join(command)}; see {log_path}")
    return elapsed, process.stdout


def generate_candidates(m, n, k, output_path):
    """Measure all CUTLASS algorithms and write the fastest ten."""
    algo_dict = torch.load(CONFIGS / "AlgoDict.pt", weights_only=True)
    algorithms = sorted((index, params) for params, index in algo_dict.items())
    overlap = torch.classes.flashoverlap_class.OverlapImpl()
    overlap.cutlass_init()
    a = torch.empty((m, k), dtype=torch.float16, device="cuda").normal_(0, 0.5)
    b = torch.empty((n, k), dtype=torch.float16, device="cuda").normal_(0, 0.5)
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    rows = []
    start = time.perf_counter()
    for algorithm, params in algorithms:
        for _ in range(10):
            overlap.cutlass_gemm(a, b, c, algorithm)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
        for begin, end in zip(starts, ends):
            begin.record()
            overlap.cutlass_gemm(a, b, c, algorithm)
            end.record()
        torch.cuda.synchronize()
        duration = statistics.mean(begin.elapsed_time(end) for begin, end in zip(starts, ends))
        rows.append((duration, algorithm, int(params[0]), int(params[1])))
    rows.sort()
    top = rows[:10]
    data = {
        "BM": [row[2] for row in top],
        "BN": [row[3] for row in top],
        "dur": [row[0] for row in top],
        "Algo": [row[1] for row in top],
    }
    output_path.write_text(json.dumps(data, indent=2))
    del a, b, c, overlap
    torch.cuda.empty_cache()
    return time.perf_counter() - start, data


def parse_test(output):
    def value(name):
        match = re.search(rf"{name} \(ms\)\s+([0-9.]+)", output)
        if not match:
            raise RuntimeError(f"missing {name} in test.py output")
        return float(match.group(1))

    return {"baseline": value("baseline_dur"), "overlap": value("overlap_dur")}


def benchmark(m, n, k, comm_op, repeats, prefix, env, log_dir):
    samples = []
    for repetition in range(repeats):
        _, output = run(
            [sys.executable, "test.py", "--m", str(m), "--n", str(n), "--k", str(k),
             "--comm_op", comm_op],
            TEST,
            env,
            log_dir / f"{prefix}_bench{repetition}.log",
        )
        samples.append(parse_test(output))
    return samples


def selected(path):
    data = json.loads(path.read_text())
    return {key: data[key] for key in ("Algo", "BM", "BN", "cSeg")}


def main():
    args = arguments()
    shapes = [parse_shape(shape) for shape in args.shapes]
    output = args.output_dir.resolve()
    logs = output / "logs"
    profiles = output / "profiles"
    chosen = output / "selected_configs"
    backups = output / "config_backups"
    for directory in (logs, profiles, chosen, backups):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    torch.ops.load_library(str(ROOT / "build/lib/libst_pybinding.so"))

    if torch.cuda.device_count() < 2:
        raise RuntimeError("at least two visible GPUs are required")
    results = []
    original_files = {}
    try:
        for m, n, k in shapes:
            label = f"{m}x{n}x{k}"
            config = CONFIGS / f"m{m}n{n}k{k}_a800.json"
            if config.exists():
                backup = backups / config.name
                shutil.copy2(config, backup)
                original_files[config] = backup

            profile = profiles / f"{label}.json"
            preprocess_s, profile_data = generate_candidates(m, n, k, profile)
            print("PREPROCESS", label, round(preprocess_s, 3), profile_data["Algo"], flush=True)

            variants = []
            for name, search_policy in (
                ("fast", "fast"),
                ("robust", "robust"),
            ):
                shutil.copy2(profile, config)
                tune_s, _ = run(
                    [sys.executable, "search.py", "--m", str(m), "--n", str(n), "--k", str(k),
                     "--comm_op", args.comm_op, "--predictive_search", "True",
                     "--search_policy", search_policy],
                    TUNE,
                    env,
                    logs / f"{label}_{name}_tune.log",
                )
                config_copy = chosen / f"{label}_{name}.json"
                shutil.copy2(config, config_copy)
                samples = benchmark(m, n, k, args.comm_op, args.repeats,
                                    f"{label}_{name}", env, logs)
                variants.append((tune_s, config_copy, samples))
                print(name.upper(), label, round(tune_s, 3), selected(config_copy), flush=True)

            fast_s, fast_config, fast_samples = variants[0]
            robust_s, robust_config, robust_samples = variants[1]

            results.append({
                "shape": label,
                "preprocess_s": preprocess_s,
                "profile_algos": profile_data["Algo"],
                "fast_tune_s": fast_s,
                "robust_tune_s": robust_s,
                "fast_config": selected(fast_config),
                "robust_config": selected(robust_config),
                "fast_bench": fast_samples,
                "robust_bench": robust_samples,
            })
            (output / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        for config, backup in original_files.items():
            shutil.copy2(backup, config)
        print("CONFIGS_RESTORED", flush=True)


if __name__ == "__main__":
    main()
