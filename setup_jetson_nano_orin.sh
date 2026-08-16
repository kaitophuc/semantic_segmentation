#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
BACKUP_DIR="${PROJECT_DIR}/venv_bad_torch"
JETSON_PYTORCH_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
CUDSS_LIB_LINE='export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu12/lib:$LD_LIBRARY_PATH"'

cd "${PROJECT_DIR}"

echo "Project: ${PROJECT_DIR}"
echo "Checking Jetson platform..."
if [[ -f /etc/nv_tegra_release ]]; then
    cat /etc/nv_tegra_release
else
    echo "Warning: /etc/nv_tegra_release not found. This script is intended for Jetson."
fi

echo
echo "Checking system TensorRT..."
if command -v trtexec >/dev/null 2>&1; then
    trtexec --help >/dev/null 2>&1 || true
else
    echo "Error: trtexec was not found. Install TensorRT from JetPack/apt first:"
    echo "  sudo apt install tensorrt python3-libnvinfer libnvinfer-bin"
    exit 1
fi

if [[ -d "${VENV_DIR}" ]]; then
    if [[ -d "${BACKUP_DIR}" ]]; then
        echo "Removing existing backup venv: ${BACKUP_DIR}"
        rm -rf "${BACKUP_DIR}"
    fi
    echo "Moving existing venv to ${BACKUP_DIR}"
    mv "${VENV_DIR}" "${BACKUP_DIR}"
fi

echo
echo "Creating venv with access to JetPack system packages..."
python3 -m venv --system-site-packages "${VENV_DIR}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo
echo "Upgrading packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing Jetson-compatible PyTorch and torchvision..."
python -m pip install --force-reinstall torch torchvision \
    --index-url "${JETSON_PYTORCH_INDEX}"

echo
echo "Installing project dependencies..."
python -m pip install -r requirements.txt

echo
echo "Installing GStreamer elements for Jetson hardware video decode..."
sudo apt-get install -y gstreamer1.0-plugins-bad gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly

echo
echo "Building fused CUDA preprocessing and overlay kernels..."
"${PROJECT_DIR}/build_cuda_kernels.sh"

if ! grep -Fq "${CUDSS_LIB_LINE}" "${VENV_DIR}/bin/activate"; then
    echo
    echo "Adding cuDSS library path to venv activation..."
    printf '\n# Jetson PyTorch cuDSS runtime library path.\n%s\n' "${CUDSS_LIB_LINE}" \
        >> "${VENV_DIR}/bin/activate"
fi

# Apply the path in this shell too, before verification.
export LD_LIBRARY_PATH="${VENV_DIR}/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"

echo
echo "Verifying Python GPU stack..."
python - <<'PY'
import cv2
import numpy
import tensorrt as trt
import torch
import transformers

print("torch:", torch.__version__, torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("tensorrt:", trt.__version__, trt.__file__)
print("cv2:", cv2.__version__)
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
PY

echo
echo "Done. Activate with:"
echo "  source venv/bin/activate"
