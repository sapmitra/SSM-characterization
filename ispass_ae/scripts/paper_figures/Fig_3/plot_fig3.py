'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-04 03:04:46
 # @ Description: Reproduce Figure 3 from the paper (Accuracy vs TTFT).
 #                See the README for usage instructions.
 '''

"""
Reproduce Figure 3 from the paper (Accuracy vs TTFT — ~1.5B models).

Reads ``ttft_logs/iteration_times.csv`` produced by ``collect_fig3_data.py`` and
``accuracy_data.csv`` (bundled in this directory) then writes two PNG files:
a clean publication figure and an annotated version.

If the TTFT CSV is not available the script falls back to the hard-coded paper
values so the figure can always be regenerated without running inference.

Usage (from repo root, any venv that has matplotlib + pandas):

    python ispass_ae/scripts/paper_figures/Fig_3/plot_fig3.py \\
        --ttft_csv    src/ttft_logs/iteration_times.csv \\
        --accuracy_csv ispass_ae/scripts/paper_figures/Fig_3/accuracy_data.csv \\
        --out_dir     ispass_ae/scripts/paper_figures/Fig_3

Output
------
accuracy_ttft.png            — publication-quality (no axis tick labels, 300 DPI)
accuracy_ttft_annotated.png  — same data with labels and TTFT values annotated
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper / fallback TTFT values  (seconds @ ~57k tokens)
# ---------------------------------------------------------------------------
PAPER_TTFT = {
    "qwen25_1.5b":    8.24,
    "mamba2_1.3b":    1.35,
    "falcon_h1_1.5b": 2.95,
}

PAPER_ACCURACY = {
    "qwen25_1.5b": {
        "MMLU": 61.13, "HellaSwag": 67.86, "Winogrande": 64.56,
        "ARC-C": 54.27,  "TruthfulQA": 47.05,
    },
    "mamba2_1.3b": {
        "MMLU": 36.3,  "HellaSwag": 59.48, "Winogrande": 58.72,
        "ARC-C": 33.2,  "TruthfulQA": 36.1,
    },
    "falcon_h1_1.5b": {
        "MMLU": 61.81, "HellaSwag": 66.76, "Winogrande": 65.59,
        "ARC-C": 53.24, "TruthfulQA": 49.39,
    },
}

MODEL_KEYS   = ["qwen25_1.5b", "mamba2_1.3b", "falcon_h1_1.5b"]
MODEL_LABELS = ["Qwen2.5-1.5B", "Mamba2-1.3B", "Falcon-H1-1.5B"]
TASKS        = ["MMLU", "HellaSwag", "Winogrande", "ARC-C", "TruthfulQA"]

BAR_COLORS  = ["#AFD2E9", "#9D96B8", "#9A7197", "#886176", "#6B4C5C"]
LINE_COLOR  = "green"

SEQ_LEN = 57344  # tokens used for TTFT measurement

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_key(name: str):
    """Map a raw model_name string from CSV to one of MODEL_KEYS.

    Only matches the specific Fig 3 model sizes:
      Qwen2.5-1.5B-Instruct, Mamba2-1.3B, Falcon-H1-1.5B-Instruct.
    Entries from other sizes (e.g. mamba2-780m, falcon-h1-0.5b) return None.
    """
    n = name.lower()
    if "qwen" in n and ("1.5b" in n or "1p5b" in n or "1_5b" in n):
        return "qwen25_1.5b"
    if "mamba2" in n and ("1.3b" in n or "1p3b" in n or "1_3b" in n):
        return "mamba2_1.3b"
    if "falcon" in n and "h1" in n and ("1.5b" in n or "1p5b" in n or "1_5b" in n):
        return "falcon_h1_1.5b"
    return None


def _load_ttft_csv(csv_path: str) -> dict:
    """Return dict model_key -> mean TTFT (seconds) for rows with seq_length == SEQ_LEN."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        result = {}
        for _, row in df.iterrows():
            key = _model_key(str(row.get("model_name", "")))
            sl  = int(row.get("seq_length", 0))
            if key is None or sl != SEQ_LEN:
                continue
            ttft = float(row.get("time_seconds", 0.0))
            result.setdefault(key, []).append(ttft)
        return {k: sum(v) / len(v) for k, v in result.items()}
    except Exception as exc:
        warnings.warn(f"Could not load TTFT CSV ({csv_path}): {exc}")
        return {}


def _load_accuracy_csv(csv_path: str) -> dict:
    """Return nested dict model_key -> {task -> accuracy}. Falls back to PAPER_ACCURACY on error."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        result = {}
        for _, row in df.iterrows():
            key  = _model_key(str(row.get("model_name", "")))
            task = str(row.get("task", ""))
            acc  = float(row.get("accuracy", 0.0))
            if key is None or task not in TASKS:
                continue
            result.setdefault(key, {})[task] = acc
        return result if result else PAPER_ACCURACY
    except Exception as exc:
        warnings.warn(f"Could not load accuracy CSV ({csv_path}): {exc}")
        return PAPER_ACCURACY


def assemble_data(ttft_csv, accuracy_csv):
    """Return (ttft_dict, accuracy_dict), falling back to paper values when CSVs are absent."""
    ttft = dict(PAPER_TTFT)
    if ttft_csv and os.path.isfile(ttft_csv):
        loaded = _load_ttft_csv(ttft_csv)
        if loaded:
            ttft.update(loaded)
            print(f"Loaded TTFT from {ttft_csv}: {loaded}")
        else:
            warnings.warn("TTFT CSV present but no matching rows found — using paper TTFT values.")
    else:
        warnings.warn("TTFT CSV not found — using hard-coded paper TTFT values.")

    accuracy = PAPER_ACCURACY
    if accuracy_csv and os.path.isfile(accuracy_csv):
        loaded_acc = _load_accuracy_csv(accuracy_csv)
        if loaded_acc:
            accuracy = loaded_acc

    return ttft, accuracy



def _build_bar_data(accuracy: dict):
    return [(task, [accuracy.get(k, {}).get(task, 0.0) for k in MODEL_KEYS]) for task in TASKS]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_annotated(ttft: dict, accuracy: dict, out_dir: str) -> None:
    """Dual-axis chart with labels — for inspection and annotated paper appendix."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    bar_data  = _build_bar_data(accuracy)
    x         = np.arange(len(MODEL_KEYS))
    bar_width = 0.15

    for i, (task, vals) in enumerate(bar_data):
        offset = (i - len(TASKS) / 2 + 0.5) * bar_width
        ax1.bar(x + offset, vals, bar_width, label=task, color=BAR_COLORS[i])

    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold", color="#003366")
    ax1.set_xlabel("Model Architecture", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_LABELS)
    ax1.set_ylim(20, 85)
    ax1.tick_params(axis="y", labelcolor="#003366")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", title="Benchmark Tasks", framealpha=0.9)

    ax2       = ax1.twinx()
    ttft_vals = [ttft[k] for k in MODEL_KEYS]
    ax2.plot(
        MODEL_LABELS, ttft_vals,
        color=LINE_COLOR, marker="o", markersize=8,
        linewidth=2.5, linestyle="--", label=f"TTFT ({SEQ_LEN // 1000}k seq)",
    )
    ax2.set_ylabel("TTFT (seconds)", fontsize=12, fontweight="bold", color=LINE_COLOR)
    ax2.tick_params(axis="y", labelcolor=LINE_COLOR)
    ax2.set_ylim(0, max(ttft_vals) * 1.3 + 1)

    for i, val in enumerate(ttft_vals):
        ax2.text(i, val + 0.3, f"{val:.2f} s", color=LINE_COLOR,
                 ha="center", fontweight="bold", fontsize=9)

    plt.title(
        f"Model Accuracy vs TTFT ({SEQ_LEN // 1000}k Sequence Length)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_ttft_annotated.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved annotated figure  ->  {path}")


def plot_paper(ttft: dict, accuracy: dict, out_dir: str) -> None:
    """Clean publication figure — no axis tick labels, 300 DPI."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    bar_data  = _build_bar_data(accuracy)
    x         = np.arange(len(MODEL_KEYS))
    bar_width = 0.15

    for i, (task, vals) in enumerate(bar_data):
        offset = (i - len(TASKS) / 2 + 0.5) * bar_width
        ax1.bar(x + offset, vals, bar_width, color=BAR_COLORS[i])

    ax1.set_ylim(20, 85)
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax1.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2       = ax1.twinx()
    ttft_vals = [ttft[k] for k in MODEL_KEYS]
    ax2.plot(
        MODEL_LABELS, ttft_vals,
        color=LINE_COLOR, marker="o", markersize=8,
        linewidth=2.5, linestyle="--",
    )
    ax2.tick_params(axis="y", which="both", left=False, right=True,
                    labelleft=False, labelright=False)
    ax2.set_ylim(0, max(ttft_vals) * 1.3 + 1)

    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_ttft.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved publication figure ->  {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Figure 3 (Accuracy vs TTFT, ~1.5B models)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ttft_csv",
        default=None,
        help="Path to src/ttft_logs/iteration_times.csv produced by collect_fig3_data.py",
    )
    p.add_argument(
        "--accuracy_csv",
        default=None,
        help="Path to accuracy_data.csv (defaults to the one bundled next to this script)",
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

    accuracy_csv = args.accuracy_csv or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "accuracy_data.csv"
    )

    ttft, accuracy = assemble_data(args.ttft_csv, accuracy_csv)

    print("\n=== Fig 3 TTFT Data ===")
    for ki, kl in zip(MODEL_KEYS, MODEL_LABELS):
        print(f"  {kl:<20} TTFT ({SEQ_LEN:>6} tokens): {ttft[ki]:.4f} s")
    print()

    plot_annotated(ttft, accuracy, args.out_dir)
    plot_paper(ttft, accuracy, args.out_dir)


if __name__ == "__main__":
    main()
