#!/usr/bin/env bash
# gen_fig6a.sh — End-to-end script to collect energy data and reproduce Figure 6a.
#
# Usage (from repo root):
#   bash ispass_ae/scripts/paper_figures/Fig_6a/gen_fig6a.sh
#
# Or from this directory:
#   bash gen_fig6a.sh
#
# The script:
#   1. Activates the Transformer venv and profiles Qwen2.5-0.5B-Instruct
#      across all 10 sequence lengths.
#   2. Activates the Mamba venv and profiles Mamba2-780m across all 10
#      sequence lengths.
#   3. Activates the Transformer venv and profiles Falcon-H1-0.5B-Base
#      across all 10 sequence lengths.
#   4. Generates the final PNG files (publication + annotated).
#
# Prerequisites:
#   - nvidia-smi must be available (NVIDIA GPU + driver).
#   - torch_transformers_ispass venv: transformers, torch, matplotlib, pandas.
#   - torch_ssm_ispass venv: mamba_ssm, transformers, torch.
#
# Output PNGs are written to the same directory as this script:
#   ispass_ae/scripts/paper_figures/Fig_6a/energy_consumption.png
#   ispass_ae/scripts/paper_figures/Fig_6a/energy_consumption_annotated.png
#
# Raw power logs: src/power_logs/
# Structured CSV:  src/energy_logs/energy_data.csv

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

# Accumulates "model@seq_len=N (exit N)" strings for failed runs.
FAILED_RUNS=()

COLLECT_SCRIPT="${SCRIPT_DIR}/collect_fig6a_data.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_fig6a.py"
ENERGY_CSV="${REPO_ROOT}/src/energy_logs/energy_data.csv"
OUT_DIR="${SCRIPT_DIR}"

TRANSFORMER_VENV="${HOME}/.venvs/torch_transformers_ispass"
FALCON_VENV="${HOME}/.venvs/torch_falcon_ispass"
MAMBA_VENV="${HOME}/.venvs/torch_ssm_ispass"

# Sequence lengths matching Figure 6a
SEQ_LENS=(1024 2048 4096 8192 16384 24576 32768 40960 49152 57344)

# ---------------------------------------------------------------------------
# Step 1a — Qwen2.5-0.5B-Instruct (Transformer venv)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1a: Qwen2.5-0.5B-Instruct — Transformer venv ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo "  [Qwen] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" --model qwen --seq_len "${SEQ_LEN}" --device cuda \
    || {
        rc=$?
        if [[ ${rc} -eq 2 ]]; then
            echo "  [SKIP] qwen seq_len=${SEQ_LEN}: OOM — continuing."
            FAILED_RUNS+=("qwen@seq_len=${SEQ_LEN} (OOM, exit ${rc})")
        else
            echo "  [SKIP] qwen seq_len=${SEQ_LEN}: unexpected error (exit ${rc}) — continuing."
            FAILED_RUNS+=("qwen@seq_len=${SEQ_LEN} (error, exit ${rc})")
        fi
    }
done

deactivate

# ---------------------------------------------------------------------------
# Step 1b — Mamba2-780m (Mamba venv)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1b: Mamba2-780m — Mamba venv ==="
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo "  [Mamba2] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" --model mamba2 --seq_len "${SEQ_LEN}" --device cuda \
    || {
        rc=$?
        if [[ ${rc} -eq 2 ]]; then
            echo "  [SKIP] mamba2 seq_len=${SEQ_LEN}: OOM — continuing."
            FAILED_RUNS+=("mamba2@seq_len=${SEQ_LEN} (OOM, exit ${rc})")
        else
            echo "  [SKIP] mamba2 seq_len=${SEQ_LEN}: unexpected error (exit ${rc}) — continuing."
            FAILED_RUNS+=("mamba2@seq_len=${SEQ_LEN} (error, exit ${rc})")
        fi
    }
done

deactivate

# ---------------------------------------------------------------------------
# Step 1c — Falcon-H1-0.5B-Base (Transformer venv)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1c: Falcon-H1-0.5B-Base — Falcon venv ==="
source "${FALCON_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo "  [Falcon-H1] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" --model falcon --seq_len "${SEQ_LEN}" --device cuda \
    || {
        rc=$?
        if [[ ${rc} -eq 2 ]]; then
            echo "  [SKIP] falcon seq_len=${SEQ_LEN}: OOM — continuing."
            FAILED_RUNS+=("falcon@seq_len=${SEQ_LEN} (OOM, exit ${rc})")
        else
            echo "  [SKIP] falcon seq_len=${SEQ_LEN}: unexpected error (exit ${rc}) — continuing."
            FAILED_RUNS+=("falcon@seq_len=${SEQ_LEN} (error, exit ${rc})")
        fi
    }
done

# ---------------------------------------------------------------------------
# Step 2 — Plot Figure 6a
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Plot Figure 6a ==="
cd "${REPO_ROOT}"

python "${PLOT_SCRIPT}" \
    --energy_csv "${ENERGY_CSV}" \
    --out_dir    "${OUT_DIR}"

deactivate

echo ""
echo "=== Done ==="
echo "Output PNGs:"
echo "  ${OUT_DIR}/energy_consumption.png"
echo "  ${OUT_DIR}/energy_consumption_annotated.png"
echo ""
echo "Raw power logs : ${REPO_ROOT}/src/power_logs/"
echo "Energy CSV     : ${ENERGY_CSV}"
echo ""
echo "To re-inspect power logs without re-running inference:"
echo "  cd ${REPO_ROOT}/src && python -m profiling.power_logger --log_dir power_logs --num_iterations 50"

# ---------------------------------------------------------------------------
# Report any skipped runs
# ---------------------------------------------------------------------------
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo "================================================================"
    echo "WARNING: ${#FAILED_RUNS[@]} run(s) were skipped due to errors:"
    for entry in "${FAILED_RUNS[@]}"; do
        echo "  - ${entry}"
    done
    echo "================================================================"
fi
