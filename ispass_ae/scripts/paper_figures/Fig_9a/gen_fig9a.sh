#!/usr/bin/env bash
# gen_fig9a.sh — End-to-end script to collect profiling data and reproduce Figure 9a.
#
# Figure 9a shows the GPU kernel time breakdown for Mamba-130m and Mamba2-130m
# across increasing sequence lengths on NVIDIA Jetson Nano.
# Sequence lengths are limited to 256–32 768 tokens (Jetson memory budget).
#
# Usage (from repo root):
#   bash ispass_ae/scripts/paper_figures/Fig_9a/gen_fig9a.sh
#
# Or from this directory:
#   bash gen_fig9a.sh
#
# The script:
#   1. Activates the SSM venv and profiles mamba-130m across all Jetson seq lengths.
#   2. Activates the SSM venv and profiles mamba2-130m across all Jetson seq lengths.
#   3. Activates the SSM venv and generates the final PNG files.
#
# Output profile CSVs are written to:
#   <REPO_ROOT>/src/profile_logs/
#
# Output PNGs are written to the same directory as this script:
#   ispass_ae/scripts/paper_figures/Fig_9a/fig9a_ops_breakdown.png
#   ispass_ae/scripts/paper_figures/Fig_9a/fig9a_ops_breakdown_annotated.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

COLLECT_SCRIPT="${SCRIPT_DIR}/collect_fig9a_data.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_fig9a.py"

# Profile data is written to src/profile_logs (shared with Fig_7)
PROFILE_DATA_DIR="${REPO_ROOT}/src/profile_logs"

OUT_DIR="${SCRIPT_DIR}"

MAMBA_VENV="${HOME}/.venvs/torch_ssm_ispass"

# Sequence lengths for Jetson Nano (up to 32 768 tokens)
SEQ_LENGTHS=(256 512 1024 2048 4096 8192 16384 32768)

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Profile mamba-130m — Mamba venv ==="
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    echo "  [mamba-130m] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" \
        --model mamba \
        --seq_len "${SEQ_LEN}" \
        --device cuda \
        --out_dir "${PROFILE_DATA_DIR}"
done

deactivate

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Profile mamba2-130m — Mamba venv ==="
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    echo "  [mamba2-130m] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" \
        --model mamba2 \
        --seq_len "${SEQ_LEN}" \
        --device cuda \
        --out_dir "${PROFILE_DATA_DIR}"
done

deactivate

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Generate Figure 9a ==="
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}"

python "${PLOT_SCRIPT}" \
    --profile_data_dir "${PROFILE_DATA_DIR}" \
    --out_dir "${OUT_DIR}"

deactivate

echo ""
echo "Done.  PNGs written to ${OUT_DIR}/"
