#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PATH="${PROJECT_DIR}/cuda/video_kernels.cu"
OUTPUT_PATH="${PROJECT_DIR}/cuda/libvideo_kernels.so"

nvcc \
    -O3 \
    --use_fast_math \
    --shared \
    -Xcompiler=-fPIC \
    -gencode arch=compute_87,code=sm_87 \
    "${SOURCE_PATH}" \
    -o "${OUTPUT_PATH}"

echo "Built ${OUTPUT_PATH}"
