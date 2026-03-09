#!/usr/bin/env bash
# gen_fig5a.sh — End-to-end script to collect memory-footprint data
#                and reproduce Figure 5a.
#
# Usage (from repo root):
#   bash ispass_ae/scripts/paper_figures/Fig_5a/gen_fig5a.sh
#
# Or from this directory:
#   bash gen_fig5a.sh
#
# Steps:
#   1. Transformer venv → profile Phi-3-mini, Qwen2.5-0.5B, Llama-3.2-1B
#   2. Mamba venv       → profile Mamba-790m, Mamba2-780m
#   3. Falcon venv      → profile Falcon-H1-0.5B and Zamba2-1.2B
#   4. Transformer venv → generate the PNG figure
#
# The CSV is written (appended) to:
#   <repo_root>/src/memory/memory_footprints.csv
#
# The output PNG is written to this directory:
#   ispass_ae/scripts/paper_figures/Fig_5a/memory_footprint_rtx_ispass.png
#
# Notes:
#   - Each model is profiled once per sequence length.
#   - Re-running appends duplicate rows; run on a fresh CSV or de-duplicate.
#   - A single sequence length can be collected with collect_fig5a_data.py
#     directly (see its --help for options).
#   - Zamba2 runs in the same `torch_falcon_ispass` venv as Falcon-H1.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

# Accumulates "model@seq_len=N (exit N)" strings for failed runs.
FAILED_RUNS=()

COLLECT_SCRIPT="${SCRIPT_DIR}/collect_fig5a_data.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_fig5a.py"
CSV_PATH="${REPO_ROOT}/src/memory/memory_footprints.csv"
OUT_DIR="${SCRIPT_DIR}"

TRANSFORMER_VENV="${HOME}/.venvs/torch_transformers_ispass"
MAMBA_VENV="${HOME}/.venvs/torch_ssm_ispass"
FALCON_VENV="${HOME}/.venvs/torch_falcon_ispass"

# ---------------------------------------------------------------------------
# Per-model sequence-length arrays (must match plotting_mem_footprint.ipynb)
# ---------------------------------------------------------------------------
PHI3_SEQ_LENS="1024 2048 4096"

QWEN_SEQ_LENS="1024 2048 4096 8192 16384 24576 32768 40960 49152 57344"

LLAMA3_2_SEQ_LENS="1024 2048 4096 8192 16384 24576 32768 40960 49152 57344 65536"

MAMBA_790M_SEQ_LENS="\
1024 2048 4096 8192 16384 24576 32768 40960 49152 57344 \
65536 81920 98304 114688 131072 147456 163840 180224 220000"

MAMBA2_780M_SEQ_LENS="\
1024 2048 4096 8192 16384 24576 32768 40960 49152 57344 \
65536 81920 98304 114688 131072 147456 163840 180224 220000"

FALCON_H1_SEQ_LENS="\
1024 2048 4096 8192 16384 24576 32768 40960 49152 57344 \
65536 81920 98304 114688 131072 147456 163840"

ZAMBA2_SEQ_LENS="1024 2048 4096 8192 16384 24576 32768 40960 49152"

# ---------------------------------------------------------------------------
# Helper: profile one model across a list of sequence lengths
#
# Exit codes returned by collect_fig5a_data.py:
#   0  — success
#   2  — out-of-memory (OOM); logged and skipped, experiment continues
#   *  — unexpected error; logged and skipped, experiment continues
# ---------------------------------------------------------------------------
profile_model() {
    local model_key="$1"
    local seq_lens="$2"
    for seq_len in $seq_lens; do
        echo "  [${model_key}] seq_len=${seq_len} ..."
        python "${COLLECT_SCRIPT}" \
            --model  "${model_key}" \
            --seq_len "${seq_len}" \
            --device cuda \
        || {
            local rc=$?
            if [[ ${rc} -eq 2 ]]; then
                echo "  [SKIP] ${model_key} seq_len=${seq_len}: OOM — continuing."
                FAILED_RUNS+=("${model_key}@seq_len=${seq_len} (OOM, exit ${rc})")
            else
                echo "  [SKIP] ${model_key} seq_len=${seq_len}: unexpected error (exit ${rc}) — continuing."
                FAILED_RUNS+=("${model_key}@seq_len=${seq_len} (error, exit ${rc})")
            fi
        }
    done
}

# ---------------------------------------------------------------------------
# Step 1 — Transformer models
# ---------------------------------------------------------------------------
echo "================================================================"
echo "Step 1: Transformer models (Phi-3-mini, Qwen2.5-0.5B, Llama-3.2-1B)"
echo "        venv: ${TRANSFORMER_VENV}"
echo "================================================================"
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

echo ""
echo "--- 1a. Phi-3-mini ---"
profile_model "phi3" "${PHI3_SEQ_LENS}"

echo ""
echo "--- 1b. Qwen2.5-0.5B-Instruct ---"
profile_model "qwen" "${QWEN_SEQ_LENS}"

echo ""
echo "--- 1c. Llama-3.2-1B-Instruct ---"
profile_model "llama3_2" "${LLAMA3_2_SEQ_LENS}"

deactivate

# ---------------------------------------------------------------------------
# Step 2 — Mamba models
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 2: Mamba models (Mamba-790m, Mamba2-780m)"
echo "        venv: ${MAMBA_VENV}"
echo "================================================================"
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

echo ""
echo "--- 2a. Mamba-790m ---"
profile_model "mamba_790m" "${MAMBA_790M_SEQ_LENS}"

echo ""
echo "--- 2b. Mamba2-780m ---"
profile_model "mamba2_780m" "${MAMBA2_780M_SEQ_LENS}"

deactivate

# ---------------------------------------------------------------------------
# Step 3 — Falcon-H1 (hybrid SSM)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 3: Falcon-H1-0.5B-Base"
echo "        venv: ${FALCON_VENV}"
echo "================================================================"
source "${FALCON_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

echo ""
echo "--- 3a. Falcon-H1-0.5B ---"
profile_model "falcon_h1" "${FALCON_H1_SEQ_LENS}"

echo ""
echo "--- 3b. Zamba2 ---"
profile_model "zamba2" "${ZAMBA2_SEQ_LENS}"

deactivate

# ---------------------------------------------------------------------------
# Step 4 — Plot
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 4: Generate Figure 5a"
echo "        venv: ${TRANSFORMER_VENV}"
echo "================================================================"
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}"

python "${PLOT_SCRIPT}" \
    --csv_path "${CSV_PATH}" \
    --out_dir  "${OUT_DIR}"

deactivate

echo ""
echo "Done.  Output:"
echo "  ${OUT_DIR}/memory_footprint_rtx_ispass.png"

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
