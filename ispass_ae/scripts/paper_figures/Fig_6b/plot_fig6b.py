'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-06
 # @ Description: Reproduce Figure 6b from the paper.  See the README for usage instructions.
 '''

"""
Reproduce Figure 6b from the paper.

Reads ``throughput_logs/generation_times.csv`` produced by
``collect_fig6b_data.py`` and generates two PNG files: a publication-quality
figure and an annotated version.

The CSV must be available; the script exits with an error if it is missing
or incomplete.

Usage (from repo root, any venv that has matplotlib + pandas):

    python ispass_ae/scripts/paper_figures/Fig_6b/plot_fig6b.py \\
        --throughput_csv src/throughput_logs/generation_times.csv \\
        --out_dir  ispass_ae/scripts/paper_figures/Fig_6b

Output files
------------
``overall_throughput_comparison.png``
    Publication-quality figure (no axis tick labels, 300 DPI).
``overall_throughput_annotated.png``
    Same data with exact values annotated on each bar (150 DPI).
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper / fallback values
# (median over measured runs, NVIDIA GPU, CUDA — from generation_times.csv)
# overall_throughput = (input_seq_length + output_tokens) / total_time_seconds
# ---------------------------------------------------------------------------
SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384, 24576, 32768]

MODEL_KEYS   = ["qwen",            "mamba2",      "falcon"]
MODEL_LABELS = ["Qwen2.5-0.5B",   "Mamba2-780m", "Falcon-H1 0.5B"]
COLORS       = ["#FF776E",         "#00C3C5",     "#809d9a"]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _model_key(name: str) -> str | None:
    """Map a raw model_name string from the CSV to 'qwen', 'mamba2', or 'falcon'."""
    name = name.lower()
    if "mamba2" in name:
        return "mamba2"
    if "qwen" in name:
        return "qwen"
    if "falcon" in name:
        return "falcon"
    return None


def _load_throughput_csv(csv_path: str) -> dict:
    """
    Load overall throughput values from ``throughput_logs/generation_times.csv``.

    Overall throughput = (input_seq_length + output_tokens) / total_time_seconds

    Returns
    -------
    dict mapping ``(model_key, input_seq_len)`` → overall_throughput (tokens/s)
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        result = {}
        valid_seq_lens = set(SEQ_LENGTHS)
        for _, row in df.iterrows():
            key = _model_key(str(row.get("model_name", "")))
            sl  = int(row.get("input_seq_length", 0))
            if key is None or sl not in valid_seq_lens:
                continue
            total_tokens    = sl + int(row.get("output_tokens", 0))
            total_time      = float(row.get("total_time_seconds", 1.0))
            overall_tp      = total_tokens / total_time if total_time > 0 else 0.0
            result[(key, sl)] = overall_tp
        return result
    except Exception as exc:
        print(f"WARNING: Could not parse throughput CSV ({csv_path}): {exc}")
        return {}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def assemble_data(throughput_csv: str | None) -> dict:
    """
    Return a dict mapping model_key → list of overall throughput values per
    seq_len (one per entry in SEQ_LENGTHS).

    Exits with an error if the CSV is missing, unreadable, or lacks required entries.
    """
    if not throughput_csv or not os.path.isfile(throughput_csv):
        print(
            f"ERROR: Throughput CSV not found: {throughput_csv!r}\n"
            "       Run collect_fig6b_data.py first, then re-run with --throughput_csv <path>."
        )
        raise SystemExit(1)

    csv_data = _load_throughput_csv(throughput_csv)
    if not csv_data:
        print(
            f"ERROR: CSV file {throughput_csv!r} could not be parsed or contains no\n"
            "       recognised (model, seq_len) entries."
        )
        raise SystemExit(1)

    missing = [
        f"  ({key}, seq_len={sl})"
        for key in MODEL_KEYS
        for sl in SEQ_LENGTHS
        if (key, sl) not in csv_data
    ]
    if missing:
        print(
            "ERROR: The following required entries are missing from the CSV:\n"
            + "\n".join(missing) + "\n"
            "       Re-run collect_fig6b_data.py to collect the missing measurements."
        )
        raise SystemExit(1)

    data: dict[str, list[float]] = {k: [] for k in MODEL_KEYS}
    for key in MODEL_KEYS:
        for sl in SEQ_LENGTHS:
            data[key].append(csv_data[(key, sl)])
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_annotated(data: dict, out_dir: str) -> None:
    """Bar chart with value labels — useful for inspection."""
    x     = np.arange(len(SEQ_LENGTHS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (key, label, color) in enumerate(zip(MODEL_KEYS, MODEL_LABELS, COLORS)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[key], width, label=label, color=color)
        for bar, val in zip(bars, data[key]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax.set_ylabel("Overall Throughput (tokens/sec)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Overall Throughput Comparison Across Sequence Lengths",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SEQ_LENGTHS, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(out_dir, "overall_throughput_annotated.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved annotated figure  →  {path}")


def plot_paper(data: dict, out_dir: str) -> None:
    """Publication-quality figure — no axis tick labels, 300 DPI."""
    x     = np.arange(len(SEQ_LENGTHS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (key, label, color) in enumerate(zip(MODEL_KEYS, MODEL_LABELS, COLORS)):
        offset = (i - 1) * width
        ax.bar(x + offset, data[key], width, label=label, color=color)

    ax.grid(axis="y", alpha=0.1, which="both")
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False,  right=False, labelleft=False)

    plt.tight_layout()
    path = os.path.join(out_dir, "overall_throughput_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication figure →  {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Figure 6b (Overall Throughput)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--throughput_csv",
        default=None,
        help="Path to src/throughput_logs/generation_times.csv produced by collect_fig6b_data.py",
    )
    p.add_argument(
        "--out_dir",
        default=".",
        help="Directory where the PNG files will be saved",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = assemble_data(args.throughput_csv)

    # Print summary table
    print("\n=== Fig 6b Data — Overall Throughput (tokens/s) ===")
    print(f"{'Seq Len':>10}", end="")
    for label in MODEL_LABELS:
        print(f"  {label:>18}", end="")
    print()
    print("-" * (10 + 20 * len(MODEL_KEYS)))
    for i, sl in enumerate(SEQ_LENGTHS):
        print(f"{sl:>10}", end="")
        for key in MODEL_KEYS:
            print(f"  {data[key][i]:>18.1f}", end="")
        print()
    print()

    plot_annotated(data, args.out_dir)
    plot_paper(data, args.out_dir)


if __name__ == "__main__":
    main()
