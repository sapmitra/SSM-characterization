'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-05
 # @ Description: Reproduce Figure 6a from the paper.  See the README for usage instructions.
 '''

"""
Reproduce Figure 6a — Prefill Energy Consumption vs Sequence Length.

Reads ``energy_logs/energy_data.csv`` produced by ``collect_fig6a_data.py``
and generates two PNG files: a publication-quality figure and an annotated
version with exact values on each bar.

If the CSV is not available (or is incomplete), the script falls back to
the hard-coded paper values so the figure can always be regenerated without
running inference.

Usage (from repo root, any venv with matplotlib + pandas):

    python ispass_ae/scripts/paper_figures/Fig_6a/plot_fig6a.py \\
        --energy_csv src/energy_logs/energy_data.csv \\
        --out_dir    ispass_ae/scripts/paper_figures/Fig_6a

Output files
------------
``energy_consumption.png``
    Publication-quality figure (no axis tick labels, 300 DPI).
``energy_consumption_annotated.png``
    Same data with exact values annotated on each bar (150 DPI).
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper / fallback values
# (measured on NVIDIA A100 40 GB, CUDA 12.4)
# Row format: seq_len → energy_joules
# ---------------------------------------------------------------------------
PAPER_VALUES = {
    "qwen": {
        1024:  7.058,
        2048:  12.85,
        4096:  24.54,
        8192:  42.22,
        16384: 98.26,
        24576: 166.97,
        32768: 512.40,
        40960: 802.99,
        49152: 1187.36,
        57344: 1492.62,
    },
    "mamba2": {
        1024:  9.881,
        2048:  14.66,
        4096:  26.45,
        8192:  53.45,
        16384: 106.02,
        24576: 158.58,
        32768: 209.18,
        40960: 261.25,
        49152: 315.32,
        57344: 370.53,
    },
    "falcon": {
        1024:  10.609,
        2048:  13.38,
        4096:  24.37,
        8192:  50.83,
        16384: 116.36,
        24576: 191.40,
        32768: 283.51,
        40960: 385.49,
        49152: 496.90,
        57344: 613.234,
    },
}

SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 57344]

MODEL_KEYS   = ["qwen",          "mamba2",       "falcon"]
MODEL_LABELS = ["Qwen2.5-0.5B",  "Mamba2-780m",  "Falcon-H1 0.5B"]
COLORS       = ["#FF776E",       "#00C3C5",       "#809d9a"]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _model_key(name: str):
    """Map a raw model_name string from the CSV to 'qwen', 'mamba2', or 'falcon'."""
    name = name.lower()
    if "mamba2" in name or "mamba-2" in name:
        return "mamba2"
    if "qwen" in name:
        return "qwen"
    if "falcon" in name:
        return "falcon"
    return None


def _load_energy_csv(csv_path: str) -> dict:
    """
    Load energy values from ``energy_logs/energy_data.csv``.

    Returns
    -------
    dict  ``{model_key: {seq_len: energy_joules}}``
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        result: dict = {"qwen": {}, "mamba2": {}, "falcon": {}}
        for _, row in df.iterrows():
            key = _model_key(str(row.get("model_name", "")))
            sl  = int(row.get("seq_len", 0))
            ej  = float(row.get("energy_joules", float("nan")))
            if key is None or sl not in SEQ_LENS:
                continue
            # Keep latest measurement for each (model, seq_len)
            result[key][sl] = ej
        return result
    except Exception as exc:
        warnings.warn(f"Could not load energy CSV ({csv_path}): {exc}")
        return {"qwen": {}, "mamba2": {}, "falcon": {}}


def _merge_with_fallback(csv_data: dict) -> dict:
    """
    For each model/seq_len, use the CSV value when available; else fall back
    to the hard-coded paper value.
    """
    merged = {}
    for mk in MODEL_KEYS:
        merged[mk] = {}
        for sl in SEQ_LENS:
            if sl in csv_data.get(mk, {}) and not _isnan(csv_data[mk][sl]):
                merged[mk][sl] = csv_data[mk][sl]
            else:
                merged[mk][sl] = PAPER_VALUES[mk][sl]
    return merged


def _isnan(v):
    try:
        import math
        return math.isnan(v)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _make_bar_plot(data: dict, annotated: bool, out_path: str):
    x = np.arange(len(SEQ_LENS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    bars = {}
    for i, (mk, label, color) in enumerate(zip(MODEL_KEYS, MODEL_LABELS, COLORS)):
        energies = [data[mk][sl] for sl in SEQ_LENS]
        offset = (i - 1) * width  # centres bars around each x tick
        b = ax.bar(x + offset, energies, width, label=label, color=color)
        bars[mk] = (b, energies)

    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.1 if not annotated else 0.3, which="both")

    if annotated:
        ax.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy Consumption (J)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Energy Consumption Comparison During Prefill Stage",
            fontsize=14, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(SEQ_LENS, rotation=45, ha="right")
        ax.legend()
        # Annotate bar values
        for mk, (b, energies) in bars.items():
            for bar, val in zip(b, energies):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 1.05,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
    else:
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    plt.tight_layout()
    dpi = 150 if annotated else 300
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce Figure 6a — prefill energy bar chart",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--energy_csv",
        default=None,
        help=(
            "Path to energy_data.csv produced by collect_fig6a_data.py "
            "(relative to cwd or absolute).  Defaults to "
            "'<repo_root>/src/energy_logs/energy_data.csv'."
        ),
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help=(
            "Directory where output PNGs are written.  "
            "Defaults to the directory of this script."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.normpath(os.path.join(script_dir, "../../../../"))

    energy_csv = args.energy_csv or os.path.join(
        repo_root, "src", "energy_logs", "energy_data.csv"
    )
    out_dir = args.out_dir or script_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load and merge
    csv_data = _load_energy_csv(energy_csv)
    data     = _merge_with_fallback(csv_data)

    n_from_csv = sum(
        1 for mk in MODEL_KEYS for sl in SEQ_LENS
        if sl in csv_data.get(mk, {}) and not _isnan(csv_data.get(mk, {}).get(sl, float("nan")))
    )
    total = len(MODEL_KEYS) * len(SEQ_LENS)
    print(f"Using {n_from_csv}/{total} data points from CSV; remainder from paper values.")

    # Publication-quality (no labels)
    _make_bar_plot(
        data, annotated=False,
        out_path=os.path.join(out_dir, "energy_consumption.png"),
    )

    # Annotated version
    _make_bar_plot(
        data, annotated=True,
        out_path=os.path.join(out_dir, "energy_consumption_annotated.png"),
    )


if __name__ == "__main__":
    main()
