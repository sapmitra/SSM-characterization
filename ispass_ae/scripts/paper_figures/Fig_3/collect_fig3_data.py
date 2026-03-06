'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-04 03:03:56
 # @ Description: Collect TTFT data for Figure 3 of the paper.  See the README for usage instructions.
 '''

"""
Collect TTFT data for Figure 3 (Accuracy vs TTFT).

Each invocation profiles one model at ~57k tokens context length 

Default model weights (higher-parameter variants, one per model choice):
  qwen25_1.5b  -> Qwen/Qwen2.5-1.5B-Instruct
  mamba2_1.3b  -> state-spaces/mamba2-1.3b
  falcon_h1_1.5b -> tiiuae/Falcon-H1-1.5B-Instruct

Usage (from ``<repo_root>/src/``):

    # --- Transformer venv -----------------------------------------------
    source ~/.venvs/torch_transformers_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_3/collect_fig3_data.py \\
        --model qwen25_1.5b --seq_len 57344 --device cuda

    # --- Mamba venv ---------------------------------------------------------
    source ~/.venvs/torch_ssm_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_3/collect_fig3_data.py \\
        --model mamba2_1.3b --seq_len 57344 --device cuda

    # --- Falcon venv --------------------------------------------------------
    source ~/.venvs/torch_falcon_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_3/collect_fig3_data.py \\
        --model falcon_h1_1.5b --seq_len 57344 --device cuda

    # Override weights (e.g. local checkpoint):
    python ../ispass_ae/scripts/paper_figures/Fig_3/collect_fig3_data.py \\
        --model qwen25_1.5b --seq_len 57344 --device cuda \\
        --weights /path/to/local/qwen25-1.5b

Profiling output is written to the ssm-characterization-out/ directory.
"""

import argparse
import sys
import os

# Allow running from src/ or from repo root: insert src/ onto sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect TTFT data for Figure 3 (Accuracy vs TTFT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["qwen25_1.5b", "mamba2_1.3b", "falcon_h1_1.5b"],
        help=(
            "Model to profile: "
            "'qwen25_1.5b' -> Qwen2.5-1.5B-Instruct, "
            "'mamba2_1.3b' -> state-spaces/mamba2-1.3b, "
            "'falcon_h1_1.5b' -> tiiuae/Falcon-H1-1.5B-Instruct"
        ),
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=57344,
        help="Prefill (input) sequence length in tokens (~57k for Fig 3)",
    )
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    p.add_argument(
        "--weights",
        default=None,
        help="Optional path / HF hub ID to override the default model weights",
    )
    return p.parse_args()


# Map each model choice to its higher-parameter default weights
_DEFAULT_WEIGHTS = {
    "qwen25_1.5b":    "Qwen/Qwen2.5-1.5B-Instruct",
    "mamba2_1.3b":    "state-spaces/mamba2-1.3b",
    "falcon_h1_1.5b": "tiiuae/Falcon-H1-1.5B-Instruct",
}


def main():
    args = parse_args()

    # Use the explicitly supplied --weights, or fall back to the higher-parameter default
    weights = args.weights or _DEFAULT_WEIGHTS[args.model]

    if args.model == "qwen25_1.5b":
        from models.profile_runner import qwen25_instruct

        print(
            f"[Fig3] Profiling Qwen2.5-1.5B-Instruct"
            f"  weights={weights}"
            f"  seq_len={args.seq_len}"
            f"  device={args.device}"
        )
        qwen25_instruct(
            seq_len=args.seq_len,
            batch_size=1,
            device=args.device,
            weights=weights,
        )

    elif args.model == "mamba2_1.3b":
        from models.profile_runner import mamba2

        print(
            f"[Fig3] Profiling state-spaces/mamba2-1.3b"
            f"  weights={weights}"
            f"  seq_len={args.seq_len}"
            f"  device={args.device}"
        )
        mamba2(
            seq_len=args.seq_len,
            batch_size=1,
            device=args.device,
            weights=weights,
        )

    else:  # falcon_h1_1.5b
        from models.profile_runner import falcon_h1

        print(
            f"[Fig3] Profiling Falcon-H1-1.5B-Instruct"
            f"  weights={weights}"
            f"  seq_len={args.seq_len}"
            f"  device={args.device}"
        )
        falcon_h1(
            seq_len=args.seq_len,
            batch_size=1,
            device=args.device,
            weights=weights,
        )


if __name__ == "__main__":
    main()
