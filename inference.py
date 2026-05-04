from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)
DEFAULT_IMAGE_SIZE = (1920, 1080)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-dir", default="models/segformer-drivable")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=DEFAULT_IMAGE_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "TensorRT engine input size. Pass one value for square input "
            "or WIDTH HEIGHT. Defaults to 1920 1080."
        ),
    )
    parser.add_argument("--output-mask", required=True)
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def read_image_size(model_dir: Path, override: Sequence[int] | None) -> tuple[int, int]:
    if override is not None:
        if len(override) == 1:
            size = int(override[0])
            return size, size
        if len(override) == 2:
            width, height = override
            return int(height), int(width)
        raise ValueError("--image-size expects one value or WIDTH HEIGHT.")

    config_path = model_dir / "preprocessor_config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    size = config["size"]
    height = int(size["height"])
    width = int(size["width"])
    return height, width


def load_engine(engine_path: Path) -> trt.ICudaEngine:
    runtime = trt.Runtime(LOGGER)
    with engine_path.open("rb") as file:
        engine = runtime.deserialize_cuda_engine(file.read())

    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")
    return engine


def torch_dtype_from_trt(dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT data type: {dtype}")
    return mapping[dtype]


def preprocess(image_path: Path, input_size: tuple[int, int]) -> torch.Tensor:
    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_resized = cv.resize(image_rgb, input_size, interpolation=cv.INTER_LINEAR)

    image_normalized = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_normalized - mean) / std

    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_transposed = np.expand_dims(image_transposed, axis=0)
    image_transposed = np.ascontiguousarray(image_transposed, dtype=np.float32)

    return image, (image.shape[1], image.shape[0]), torch.from_numpy(image_transposed)
