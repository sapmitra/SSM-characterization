#!/usr/bin/env bash
# gen_fig9b.sh — End-to-end script to collect profiling data and reproduce Figure 9b.
#
# Figure 9b shows the GPU kernel-time breakdown for ALL model families
# (Transformer, SSM, Hybrid) at a fixed sequence length (default 1024 tokens)
# on two hardware platforms: Desktop GPU and NVIDIA Jetson Orin Nano.
#
# ─── Quick start ────────────────────────────────────────────────────────────
# Desktop only (profiles + plots in one step):
#   bash ispass_ae/scripts/paper_figures/Fig_9b/gen_fig9b.sh
#   # Or from this directory: bash gen_fig9b.sh
# ────────────────────────────────────────────────────────────────────────────
#
# ─── Two-device workflow ────────────────────────────────────────────────────
# Step A — Desktop (collect + plot once Jetson data is merged):
#   bash gen_fig9b.sh
#   # Profiles all models → src/profile_logs/
#   # Venvs resolved from ~/.venvs/  (auto-detected)
#
# Step B — Jetson (collect only, skip the plot):
#   SKIP_PLOT=1 bash gen_fig9b.sh
#   # Profiles all models → src/profile_logs/
#   # Venvs resolved from /data/.venvs/  (auto-detected via /etc/nv_tegra_release)
#   # Override: IS_JETSON=1 bash gen_fig9b.sh  (force Jetson venv paths)
#
# Step C — Transfer Jetson data to workstation:
#   rsync -avz jetson:~/path/to/repo/src/profile_logs/ \
#             ~/path/to/repo/src/profile_logs_jetson/
#
# Step D — Plot on workstation (after transfer):
#   source ~/.venvs/torch_transformers_ispass/bin/activate
#   cd <repo_root>
#   python ispass_ae/scripts/paper_figures/Fig_9b/plot_fig9b.py
# ────────────────────────────────────────────────────────────────────────────
#
# The script:
#   1. Activates the Transformer venv and profiles Transformer models.
#   2. Activates the SSM venv and profiles Mamba-130m and Mamba2-130m.
#   3. Activates the Hybrid venv and profiles Hymba and Zamba2.
#   4. Activates the Transformer venv and generates the comparison PNG
#      (skipped when SKIP_PLOT=1).
#
# Output profile CSVs are written to:
#   <REPO_ROOT>/src/profile_logs/        (default, override via PROFILE_DATA_DIR)
#
# Output PNG is written to:
#   ispass_ae/scripts/paper_figures/Fig_9b/fig9b_device_comparison_seq<N>.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

COLLECT_SCRIPT="${SCRIPT_DIR}/collect_fig9b_data.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_fig9b.py"

# ---------------------------------------------------------------------------
# Configuration — override via environment variables if needed.
# ---------------------------------------------------------------------------

# Profile data destination.  Override to change output path (e.g. on Jetson).
PROFILE_DATA_DIR="${PROFILE_DATA_DIR:-${REPO_ROOT}/src/profile_logs}"

# Jetson data directory (used during the plot step on the workstation).
JETSON_DATA_DIR="${JETSON_DATA_DIR:-${REPO_ROOT}/src/profile_logs_jetson}"

# Sequence length for the cross-device breakdown comparison.
SEQ_LEN="${SEQ_LEN:-1024}"

# Set SKIP_PLOT=1 on the Jetson (or when the Jetson data is not yet available).
SKIP_PLOT="${SKIP_PLOT:-0}"

# Output directory (PNGs are saved here).
OUT_DIR="${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Jetson detection — presence of /etc/nv_tegra_release identifies a Jetson board.
# Override with IS_JETSON=1 or IS_JETSON=0 to force a specific behaviour.
# On Jetson the venvs live under /data/.venvs/ (larger storage partition);
# on a desktop workstation they live under ~/.venvs/.
# ---------------------------------------------------------------------------
if [[ -z "${IS_JETSON:-}" ]]; then
    if [[ -f /etc/nv_tegra_release ]]; then
        IS_JETSON=1
    else
        IS_JETSON=0
    fi
fi

if [[ "${IS_JETSON}" == "1" ]]; then
    VENV_BASE="/data/.venvs"
else
    VENV_BASE="${HOME}/.venvs"
fi

# Virtual environments
TRANSFORMER_VENV="${VENV_BASE}/torch_transformers_ispass"
MAMBA_VENV="${VENV_BASE}/torch_ssm_ispass"
FALCON_VENV="${VENV_BASE}/torch_falcon_ispass"

mkdir -p "${PROFILE_DATA_DIR}"

echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  Fig 9b — Cross-Device Operator Breakdown                    │"
echo "│  seq_len        : ${SEQ_LEN}"
echo "│  profile_data   : ${PROFILE_DATA_DIR}"
echo "│  skip_plot      : ${SKIP_PLOT}"
echo "│  is_jetson      : ${IS_JETSON}  (venvs: ${VENV_BASE})"
echo "└──────────────────────────────────────────────────────────────┘"

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Profile Transformer models — Transformer venv ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for MODEL in gpt-neo-125m tinyllama llama3_2 qwen25-instruct qwen25-1.5b-instruct; do
    echo "  [${MODEL}] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" \
        --model "${MODEL}" \
        --seq_len "${SEQ_LEN}" \
        --device cuda \
        --out_dir "${PROFILE_DATA_DIR}"
done

deactivate

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Profile SSM models — Mamba venv ==="
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for MODEL in mamba mamba2; do
    echo "  [${MODEL}] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" \
        --model "${MODEL}" \
        --seq_len "${SEQ_LEN}" \
        --device cuda \
        --out_dir "${PROFILE_DATA_DIR}"
done

deactivate

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Profile Hybrid models — Falcon/Hybrid venv ==="
source "${FALCON_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for MODEL in hymba zamba2; do
    echo "  [${MODEL}] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" \
        --model "${MODEL}" \
        --seq_len "${SEQ_LEN}" \
        --device cuda \
        --out_dir "${PROFILE_DATA_DIR}"
done

deactivate

# ---------------------------------------------------------------------------
if [[ "${SKIP_PLOT}" == "1" ]]; then
    echo ""
    echo "=== SKIP_PLOT=1: skipping plot step ==="
    echo ""
    echo "Data collection complete.  Profile CSVs written to:"
    echo "  ${PROFILE_DATA_DIR}"
    echo ""
    echo "Transfer these to the workstation and re-run with:"
    echo "  rsync -avz <jetson>:${PROFILE_DATA_DIR}/ <workstation>:src/profile_logs_jetson/"
    echo "  python ${PLOT_SCRIPT}"
    exit 0
fi

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Generate Figure 9b ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}"

python "${PLOT_SCRIPT}" \
    --desktop_dir "${PROFILE_DATA_DIR}" \
    --jetson_dir  "${JETSON_DATA_DIR}" \
    --seq_len     "${SEQ_LEN}" \
    --out_dir     "${OUT_DIR}"

deactivate

echo ""
echo "Done.  PNG written to ${OUT_DIR}/fig9b_device_comparison_seq${SEQ_LEN}.png"
