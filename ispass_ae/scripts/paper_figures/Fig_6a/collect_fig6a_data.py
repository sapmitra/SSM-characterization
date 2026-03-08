'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-05
 # @ Description: Collect prefill-phase energy data for Figure 6a of the paper.
 #                See the README for usage instructions.
 '''

"""
Collect prefill-phase energy consumption (Joules) for Figure 6a.

Each invocation profiles one model at one sequence length and appends a row to
``src/energy_logs/energy_data.csv``.  Run all invocations (all models × all
sequence lengths) before plotting.

This script should be invoked from ``<repo_root>/src/`` with the appropriate
virtual environment active.

Usage (from ``<repo_root>/src/``):

    # --- Transformer venv (Qwen2.5-0.5B-Instruct) ---------------------------
    source ~/.venvs/torch_transformers_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_6a/collect_fig6a_data.py \\
        --model qwen --seq_len 1024 --device cuda

    # --- Falcon venv (Falcon-H1-0.5B) ----------------------------------------
    source ~/.venvs/torch_falcon_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_6a/collect_fig6a_data.py \\
        --model falcon --seq_len 1024 --device cuda

    # --- Mamba venv (Mamba2-780m) ------------------------------------------
    source ~/.venvs/torch_ssm_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_6a/collect_fig6a_data.py \\
        --model mamba2 --seq_len 1024 --device cuda

Output CSV (relative to where the script is invoked, i.e. ``src/``):

    energy_logs/energy_data.csv
        model_name, seq_len, energy_joules, duration_per_pass_s,
        avg_power_watts, num_iterations, device, timestamp

    power_logs/<model_name>_power.log
        Raw nvidia-smi CSV (100 ms samples) — parsed internally by
        ``profiling/power_logger.py::parse_energy_from_log()``.

Sequence lengths used in Figure 6a:
    1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 57344
"""

import argparse
import os
import sys

# Allow running from src/ or from repo root
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


# Sequence lengths covering Figure 6a
SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 57344]


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect prefill energy data for Figure 6a",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["qwen", "mamba2", "falcon"],
        help=(
            "Model to profile: "
            "'qwen' → Qwen2.5-0.5B-Instruct, "
            "'mamba2' → Mamba2-780m, "
            "'falcon' → Falcon-H1-0.5B-Base"
        ),
    )
    p.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help="Prefill (input) sequence length in tokens",
    )
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    p.add_argument(
        "--weights",
        default=None,
        help="Optional HuggingFace repo ID or local path to override default weights",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.model == "qwen":
        from models.profile_runner import qwen25_instruct_energy

        print(
            f"[Fig6a] Profiling Qwen2.5-0.5B-Instruct (energy)"
            f"  seq_len={args.seq_len}"
            f"  device={args.device}"
        )
        qwen25_instruct_energy(
            seq_len=args.seq_len,
            device=args.device,
            weights=args.weights,
        )

    elif args.model == "mamba2":
        from models.profile_runner import mamba2_energy

        print(
            f"[Fig6a] Profiling Mamba2-780m (energy)"
            f"  seq_len={args.seq_len}"
            f"  device={args.device}"
        )
        mamba2_energy(
            seq_len=args.seq_len,
            device=args.device,
            weights=args.weights,
        )

    else:  # falcon
        from models.profile_runner import falcon_h1_energy

        print(
            f"[Fig6a] Profiling Falcon-H1-0.5B-Base (energy)"
            f"  seq_len={args.seq_len}"
            f"  device={args.device}"
        )
        falcon_h1_energy(
            seq_len=args.seq_len,
            device=args.device,
            weights=args.weights,
        )


if __name__ == "__main__":
    main()
