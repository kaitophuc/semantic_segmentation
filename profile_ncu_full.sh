#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${PROJECT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [[ ${EUID} -ne 0 ]]; then
    exec sudo "${SCRIPT_PATH}" "$@"
fi

NCU_BIN="/usr/local/cuda-12.6/bin/ncu"
PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
CUDSS_LIB_DIR="${PROJECT_DIR}/venv/lib/python3.10/site-packages/nvidia/cu12/lib"
INPUT_IMAGE="${PROJECT_DIR}/Data/output/IMG_3767_frame0.jpg"
ENGINE_PATH="${PROJECT_DIR}/models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine"
MODEL_DIR="${PROJECT_DIR}/models/segformer-drivable"
REPORT_BASE="${PROJECT_DIR}/Data/output/ncu_full_all_kernels_single_frame"
REPORT_PATH="${REPORT_BASE}.ncu-rep"

for required_path in \
    "${NCU_BIN}" \
    "${PYTHON_BIN}" \
    "${INPUT_IMAGE}" \
    "${ENGINE_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Error: required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

export LD_LIBRARY_PATH="${CUDSS_LIB_DIR}:${LD_LIBRARY_PATH:-}"
cd "${PROJECT_DIR}"

"${NCU_BIN}" \
    --set full \
    --replay-mode application \
    --app-replay-match grid \
    --app-replay-mode strict \
    --target-processes application-only \
    --force-overwrite \
    --export "${REPORT_BASE}" \
    "${PYTHON_BIN}" inference.py \
        --image "${INPUT_IMAGE}" \
        --engine "${ENGINE_PATH}" \
        --model-dir "${MODEL_DIR}"

"${NCU_BIN}" --import "${REPORT_PATH}" --page session >/dev/null

if [[ -n "${SUDO_UID:-}" && -n "${SUDO_GID:-}" ]]; then
    chown "${SUDO_UID}:${SUDO_GID}" "${REPORT_PATH}" || true
fi

echo "Validated NCU report: ${REPORT_PATH}"
