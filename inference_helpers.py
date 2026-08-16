"""Small, stateless helpers shared by the inference workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np

if TYPE_CHECKING:
    import tensorrt as trt


DEFAULT_INPUT_VIDEO = Path("Data/input/IMG_3767.MOV")
DEFAULT_OUTPUT_VIDEO = Path("Data/output/IMG_3767_overlay.mp4")
DEFAULT_CUDA_KERNELS = Path("cuda/libvideo_kernels.so")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SCALE = (1.0 / (255.0 * IMAGENET_STD)).astype(np.float32)
INPUT_BIAS = (-IMAGENET_MEAN / IMAGENET_STD).astype(np.float32)
BGR_TO_RGB_CHANNELS = (2, 1, 0)
PIXEL_VALUES = np.arange(256, dtype=np.float32)[:, None]
INPUT_LOOKUP_FLOAT16 = (
    (PIXEL_VALUES / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
).T.astype(np.float16)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int


def parse_image_size(override: Sequence[int]) -> tuple[int, int]:
    """Convert CLI WIDTH/HEIGHT values to the internal (height, width) order."""
    if len(override) == 1:
        size = int(override[0])
        return size, size
    if len(override) == 2:
        width, height = override
        return int(height), int(width)
    raise ValueError("--image-size expects one value or WIDTH HEIGHT.")


def read_processor_size(model_dir: Path) -> tuple[int, int]:
    config_path = model_dir / "preprocessor_config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    size = config["size"]
    return int(size["height"]), int(size["width"])


def check_cuda_ready() -> None:
    # Import lazily so the TensorRT runtime module can initialize JetPack's
    # shared libraries before PyTorch is loaded.
    import torch

    if torch.cuda.is_available():
        return

    raise RuntimeError(
        "CUDA is not available to PyTorch, so TensorRT cannot run either. "
        "On Jetson, make sure PyTorch/TensorRT come from the JetPack-compatible "
        "NVIDIA packages and that the GPU device nodes are available."
    )


def choose_input_size(
    engine: trt.ICudaEngine,
    input_name: str,
    model_dir: Path,
    override: Sequence[int] | None,
) -> tuple[int, int]:
    if override is not None:
        return parse_image_size(override)

    engine_shape = tuple(engine.get_tensor_shape(input_name))
    if len(engine_shape) == 4 and engine_shape[2] > 0 and engine_shape[3] > 0:
        return int(engine_shape[2]), int(engine_shape[3])

    return read_processor_size(model_dir)


def preprocess_frame(
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],
    *,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Resize and normalize a BGR frame into RGB NCHW model input."""
    input_h, input_w = input_size
    expected_shape = (1, 3, input_h, input_w)

    if output is None:
        output = np.empty(expected_shape, dtype=np.float32)
    elif output.shape != expected_shape:
        raise ValueError(
            f"Preprocessing output shape {output.shape} does not match {expected_shape}"
        )
    elif output.dtype not in (np.float16, np.float32):
        raise ValueError(
            "Preprocessing output must use float16 or float32, "
            f"found {output.dtype}"
        )

    if frame_bgr.shape[:2] == (input_h, input_w):
        frame_resized = frame_bgr
    else:
        frame_resized = cv.resize(
            frame_bgr,
            (input_w, input_h),
            interpolation=cv.INTER_LINEAR,
        )

    if output.dtype == np.float16:
        # Match the legacy float32-normalize-then-cast behavior for FP16 bindings.
        for output_channel, input_channel in enumerate(BGR_TO_RGB_CHANNELS):
            np.take(
                INPUT_LOOKUP_FLOAT16[output_channel],
                frame_resized[:, :, input_channel],
                out=output[0, output_channel],
            )
    else:
        # Write BGR source channels directly into normalized RGB CHW planes.
        for output_channel, input_channel in enumerate(BGR_TO_RGB_CHANNELS):
            channel_output = output[0, output_channel]
            np.multiply(
                frame_resized[:, :, input_channel],
                INPUT_SCALE[output_channel],
                out=channel_output,
            )
            np.add(
                channel_output,
                INPUT_BIAS[output_channel],
                out=channel_output,
            )

    return output


def postprocess(logits: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """Turn binary logits into a full-resolution uint8 class mask."""
    mask_small = np.argmax(logits, axis=1)[0].astype(np.uint8)
    original_h, original_w = original_size
    return cv.resize(mask_small, (original_w, original_h), interpolation=cv.INTER_NEAREST)


def make_overlay(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    green = np.zeros_like(frame_bgr)
    green[:, :, 1] = 255

    blended = cv.addWeighted(frame_bgr, 1.0 - alpha, green, alpha, 0)
    overlay = frame_bgr.copy()
    overlay[mask == 1] = blended[mask == 1]
    return overlay


def probe_video(video_path: Path) -> VideoInfo:
    """Read the metadata used to size and time the video pipeline."""
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video not found or could not be opened: {video_path}")
    try:
        fps = capture.get(cv.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        return VideoInfo(
            width=int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            total_frames=int(capture.get(cv.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()
