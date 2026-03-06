#!/usr/bin/env bash
# gen_fig3.sh — End-to-end script to collect TTFT data and reproduce Figure 3.
#
# Usage (from repo root):
#   bash ispass_ae/scripts/paper_figures/Fig_3/gen_fig3.sh
#
# Or from this directory:
#   bash gen_fig3.sh
#
# Steps:
#   1. Activate Transformer venv  → profile Qwen2.5-1.5B at ~57k tokens
#   2. Activate Mamba venv        → profile Mamba2-1.3b  at ~57k tokens
#   3. Activate Falcon venv       → profile Falcon-H1-1.5B at ~57k tokens
#   4. Activate Transformer venv  → plot the figure
#
# Output PNGs are written to:
#   ispass_ae/scripts/paper_figures/Fig_3/accuracy_ttft.png
#   ispass_ae/scripts/paper_figures/Fig_3/accuracy_ttft_annotated.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

COLLECT_SCRIPT="${SCRIPT_DIR}/collect_fig3_data.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_fig3.py"
TTFT_CSV="${REPO_ROOT}/src/ttft_logs/iteration_times.csv"
ACCURACY_CSV="${SCRIPT_DIR}/accuracy_data.csv"
OUT_DIR="${SCRIPT_DIR}"

TRANSFORMER_VENV="${HOME}/.venvs/torch_transformers_ispass"
MAMBA_VENV="${HOME}/.venvs/torch_ssm_ispass"
FALCON_VENV="${HOME}/.venvs/torch_falcon_ispass"

SEQ_LEN=57344

echo "=== Step 1: Qwen2.5-1.5B — Transformer venv ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}/src"
echo "  [Qwen2.5-1.5B] seq_len=${SEQ_LEN} ..."
python "${COLLECT_SCRIPT}" --model qwen25_1.5b --seq_len ${SEQ_LEN} --device cuda
deactivate

echo ""
echo "=== Step 2: Mamba2-1.3b — Mamba venv ==="
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}/src"
echo "  [Mamba2-1.3b] seq_len=${SEQ_LEN} ..."
python "${COLLECT_SCRIPT}" --model mamba2_1.3b --seq_len ${SEQ_LEN} --device cuda
deactivate

echo ""
echo "=== Step 3: Falcon-H1-1.5B — Falcon venv ==="
source "${FALCON_VENV}/bin/activate"
cd "${REPO_ROOT}/src"
echo "  [Falcon-H1-1.5B] seq_len=${SEQ_LEN} ..."
python "${COLLECT_SCRIPT}" --model falcon_h1_1.5b --seq_len ${SEQ_LEN} --device cuda
deactivate

echo ""
echo "=== Step 4: Plot Figure 3 ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}"
python "${PLOT_SCRIPT}" \
    --ttft_csv     "${TTFT_CSV}" \
    --accuracy_csv "${ACCURACY_CSV}" \
    --out_dir      "${OUT_DIR}"
deactivate

echo ""
echo "Done. Output files:"
echo "  ${OUT_DIR}/accuracy_ttft.png"
echo "  ${OUT_DIR}/accuracy_ttft_annotated.png"
