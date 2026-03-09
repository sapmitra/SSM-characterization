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

The CSV must be available; the script exits with an error if it is missing
or incomplete.

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
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        print(f"WARNING: Could not parse energy CSV ({csv_path}): {exc}")
        return {"qwen": {}, "mamba2": {}, "falcon": {}}


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

    # Load and validate
    if not os.path.isfile(energy_csv):
        print(
            f"ERROR: Energy CSV not found: {energy_csv!r}\n"
            "       Run collect_fig6a_data.py first, then re-run with --energy_csv <path>."
        )
        sys.exit(1)

    csv_data = _load_energy_csv(energy_csv)

    missing = [
        f"  ({mk}, seq_len={sl})"
        for mk in MODEL_KEYS
        for sl in SEQ_LENS
        if sl not in csv_data.get(mk, {})
    ]
    if missing:
        print(
            "ERROR: The following required entries are missing from the CSV:\n"
            + "\n".join(missing) + "\n"
            "       Re-run collect_fig6a_data.py to collect the missing measurements."
        )
        sys.exit(1)

    # Build data dict from CSV values only
    data: dict = {mk: {sl: csv_data[mk][sl] for sl in SEQ_LENS} for mk in MODEL_KEYS}
    print(f"Loaded {len(MODEL_KEYS) * len(SEQ_LENS)} data points from {energy_csv}.")

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
