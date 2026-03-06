'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-02 03:06:18
 # @ Description: Parses nvidia-smi power log files (power_logs/*.log) and
 #                 computes per-GPU average power draw and energy consumption.
 #                 Exposes parse_energy_from_log() as a reusable public API used
 #                 by profiling/eval.py.
 '''

import os
import math
import glob
import argparse
import pandas


# nvidia-smi is invoked with --loop-ms=100, so each sample covers 0.1 s
SAMPLE_INTERVAL_S = 0.1

_NUMERIC_COLS = [
    " power.draw [W]",
    " memory.used [MiB]",
    " utilization.memory [%]",
    " utilization.gpu [%]",
]


def _clean_numeric(column):
    """Strip unit suffixes (e.g. ' W', ' MiB', ' %') and return float Series."""
    return column.str.replace(r"[^\d.]", "", regex=True).astype(float)


def parse_energy_from_log(log_file: str, num_iterations: int) -> dict:
    """Parse an nvidia-smi CSV power log and compute per-pass GPU energy.

    Parameters
    ----------
    log_file : str
        Path to the CSV log produced by nvidia-smi --format=csv --loop-ms=100.
    num_iterations : int
        Number of inference passes that were recorded (used to normalise energy
        so the returned value is *per-pass* energy in Joules).

    Returns
    -------
    dict with keys:
        gpu{i}_energy_joules   - per-pass energy for GPU i (J)
        gpu{i}_avg_power_watts - mean power draw for GPU i (W)
        total_energy_joules    - sum of per-pass energy across all active GPUs (J)
        avg_power_watts        - mean power draw across all active GPUs (W)
    """
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be > 0, got {num_iterations}")

    df = pandas.read_csv(log_file)

    # Clean numeric columns (nvidia-smi appends unit strings)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    result = {}
    gpu_ids = sorted(df["index"].dropna().unique().astype(int))

    active_gpus = []
    for gid in gpu_ids:
        gpu_df = df[df["index"] == gid]
        power_col = gpu_df[" power.draw [W]"].dropna()
        if power_col.empty:
            continue

        energy_j = power_col.sum() * SAMPLE_INTERVAL_S / num_iterations
        avg_power_w = power_col.mean()

        if math.isnan(energy_j) or math.isnan(avg_power_w):
            continue

        result[f"gpu{gid}_energy_joules"] = energy_j
        result[f"gpu{gid}_avg_power_watts"] = avg_power_w
        active_gpus.append(gid)

    if active_gpus:
        result["total_energy_joules"] = sum(
            result[f"gpu{g}_energy_joules"] for g in active_gpus
        )
        result["avg_power_watts"] = sum(
            result[f"gpu{g}_avg_power_watts"] for g in active_gpus
        ) / len(active_gpus)
    else:
        result["total_energy_joules"] = 0.0
        result["avg_power_watts"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Script mode - batch-process all *.log files in a directory
# ---------------------------------------------------------------------------

def _process_all_logs(log_dir: str, num_iterations: int) -> None:
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not log_files:
        print(f"No .log files found in {log_dir}")
        return

    for log_file in log_files:
        filename = os.path.basename(log_file)
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"{'='*80}")

        try:
            energy = parse_energy_from_log(log_file, num_iterations)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        for key, val in sorted(energy.items()):
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse nvidia-smi power logs and compute per-pass GPU energy."
    )
    parser.add_argument(
        "--log_dir",
        default="power_logs",
        help="Directory containing *.log files (default: power_logs)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Number of inference passes recorded in each log (default: 50)",
    )
    args = parser.parse_args()
    _process_all_logs(args.log_dir, args.num_iterations)
