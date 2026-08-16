#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
MODEL_DIR="${PROJECT_DIR}/models/segformer-drivable"
ONNX_PATH="${MODEL_DIR}/segformer_drivable_1920x1080_stride8.onnx"
TARGET_ENGINE="${MODEL_DIR}/segformer_drivable_1920x1080_fp16.engine"
OPTIMIZED_ENGINE="${MODEL_DIR}/segformer_drivable_1920x1080_stride8_fp16.engine"
BASELINE_ENGINE="${MODEL_DIR}/segformer_drivable_1920x1080_stride4_baseline.engine"
NEW_ENGINE="${MODEL_DIR}/segformer_drivable_1920x1080_stride8_fp16.engine.new"

for required_path in "${VENV_PYTHON}" "${TRTEXEC}"; do
    if [[ ! -x "${required_path}" ]]; then
        echo "Error: required executable does not exist: ${required_path}" >&2
        exit 1
    fi
done

cd "${PROJECT_DIR}"

./build_cuda_kernels.sh

"${VENV_PYTHON}" export_onnx.py \
    --model-dir "${MODEL_DIR}" \
    --output "${ONNX_PATH}" \
    --image-size 1920 1080 \
    --first-patch-stride 8 \
    --device cpu

if [[ -f "${TARGET_ENGINE}" && ! -f "${BASELINE_ENGINE}" ]]; then
    cp --preserve=mode,timestamps "${TARGET_ENGINE}" "${BASELINE_ENGINE}"
    echo "Saved the original stride-4 engine to ${BASELINE_ENGINE}"
fi

"${TRTEXEC}" \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${NEW_ENGINE}" \
    --fp16 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw \
    --builderOptimizationLevel=3 \
    --maxAuxStreams=0 \
    --profilingVerbosity=detailed \
    --skipInference

cp "${NEW_ENGINE}" "${OPTIMIZED_ENGINE}"
mv "${NEW_ENGINE}" "${TARGET_ENGINE}"

"${TRTEXEC}" \
    --loadEngine="${TARGET_ENGINE}" \
    --warmUp=1000 \
    --duration=3 \
    --iterations=100 \
    --noDataTransfers \
    --useCudaGraph

echo "Installed optimized engine at ${TARGET_ENGINE}"
echo "The engine input remains 1x3x1080x1920."
