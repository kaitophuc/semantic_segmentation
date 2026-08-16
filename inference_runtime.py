"""Persistent TensorRT execution and fused CUDA video processing."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import sys


# JetPack installs TensorRT outside most virtual environments. Temporarily make
# that location importable without leaving it ahead of the venv on sys.path.
SYSTEM_TENSORRT_PATH = Path("/usr/lib/python3.10/dist-packages")
if SYSTEM_TENSORRT_PATH.exists() and str(SYSTEM_TENSORRT_PATH) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TENSORRT_PATH))

import tensorrt as trt

if sys.path[0] == str(SYSTEM_TENSORRT_PATH):
    sys.path.pop(0)

import numpy as np
import torch


LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path: Path) -> trt.ICudaEngine:
    runtime = trt.Runtime(LOGGER)
    with engine_path.open("rb") as file:
        engine = runtime.deserialize_cuda_engine(file.read())

    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")
    return engine


def _torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
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


def _get_io_names(engine: trt.ICudaEngine) -> tuple[str, str]:
    input_names = []
    output_names = []

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
        else:
            raise ValueError(f"Unknown tensor mode {mode} for tensor {name}")

    if len(input_names) != 1:
        raise ValueError(f"Expected exactly one input tensor, found {len(input_names)}")
    if len(output_names) != 1:
        raise ValueError(f"Expected exactly one output tensor, found {len(output_names)}")

    return input_names[0], output_names[0]


class CudaVideoKernels:
    """Thin ctypes wrapper around the fused kernels in cuda/video_kernels.cu."""

    def __init__(self, library_path: Path) -> None:
        if not library_path.exists():
            raise FileNotFoundError(
                f"Fused CUDA library not found: {library_path}. "
                "Run ./build_cuda_kernels.sh first."
            )

        self.library = ctypes.CDLL(str(library_path.resolve()))
        self.library.preprocess_bgr_to_rgb_chw.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        self.library.preprocess_bgr_to_rgb_chw.restype = ctypes.c_int
        self.library.overlay_binary_logits.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        )
        self.library.overlay_binary_logits.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"CUDA {operation} kernel launch failed with status {status}")

    def preprocess(
        self,
        device_frame: torch.Tensor,
        device_input: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        height, width, channels = device_frame.shape
        status = self.library.preprocess_bgr_to_rgb_chw(
            ctypes.c_void_p(device_frame.data_ptr()),
            ctypes.c_void_p(device_input.data_ptr()),
            height,
            width,
            channels,
            int(device_input.dtype == torch.float16),
            ctypes.c_void_p(stream.cuda_stream),
        )
        self._check(status, "preprocessing")

    def overlay(
        self,
        device_frame: torch.Tensor,
        device_logits: torch.Tensor,
        device_overlay: torch.Tensor,
        alpha: float,
        stream: torch.cuda.Stream,
    ) -> None:
        frame_height, frame_width, channels = device_frame.shape
        if (
            device_logits.ndim != 4
            or device_logits.shape[0] != 1
            or device_logits.shape[1] != 2
        ):
            raise ValueError(
                "Fused overlay expects binary logits shaped (1, 2, H, W), "
                f"found {tuple(device_logits.shape)}"
            )
        logits_height = int(device_logits.shape[2])
        logits_width = int(device_logits.shape[3])
        status = self.library.overlay_binary_logits(
            ctypes.c_void_p(device_frame.data_ptr()),
            ctypes.c_void_p(device_logits.data_ptr()),
            ctypes.c_void_p(device_overlay.data_ptr()),
            frame_height,
            frame_width,
            channels,
            logits_height,
            logits_width,
            int(device_logits.dtype == torch.float16),
            ctypes.c_float(alpha),
            ctypes.c_void_p(stream.cuda_stream),
        )
        self._check(status, "overlay")


@dataclass
class VideoPipelineSlot:
    """Pinned host buffers passed between decode, inference, and encode."""

    host_frame: torch.Tensor
    host_frame_array: np.ndarray
    host_overlay: torch.Tensor
    host_overlay_array: np.ndarray
    frame_index: int = -1


class TensorRTRunner:
    """One persistent TensorRT context, stream, and set of model buffers."""

    def __init__(self, engine: trt.ICudaEngine) -> None:
        self.engine = engine
        self.context = engine.create_execution_context()
        self.input_name, self.output_name = _get_io_names(engine)
        self.input_dtype = _torch_dtype_from_trt(
            engine.get_tensor_dtype(self.input_name)
        )
        self.output_dtype = _torch_dtype_from_trt(
            engine.get_tensor_dtype(self.output_name)
        )
        self.inference_stream = torch.cuda.Stream()
        self.input_shape: tuple[int, ...] | None = None
        self.output_shape: tuple[int, ...] | None = None
        self.host_input: torch.Tensor | None = None
        self.host_input_array: np.ndarray | None = None
        self.device_input: torch.Tensor | None = None
        self.device_output: torch.Tensor | None = None
        self.host_output: torch.Tensor | None = None
        self.device_frame: torch.Tensor | None = None
        self.device_overlay: torch.Tensor | None = None
        self.cuda_video_kernels: CudaVideoKernels | None = None

    def _configure_context(
        self,
        context: trt.IExecutionContext,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if -1 in engine_input_shape:
            context.set_input_shape(self.input_name, input_shape)
        elif input_shape != engine_input_shape:
            raise ValueError(
                f"Input shape {input_shape} does not match engine shape {engine_input_shape}"
            )

        output_shape = tuple(context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"TensorRT output shape is still dynamic: {output_shape}")
        return output_shape

    def configure(self, input_shape: tuple[int, ...]) -> None:
        """Allocate persistent buffers for the runner's one supported shape."""
        if self.input_shape == input_shape:
            return
        if self.input_shape is not None:
            raise RuntimeError(
                f"Runner is already configured for {self.input_shape}, not {input_shape}"
            )

        output_shape = self._configure_context(self.context, input_shape)
        self.host_input = torch.empty(
            input_shape, dtype=self.input_dtype, device="cpu", pin_memory=True
        )
        self.host_input_array = self.host_input.numpy()
        self.device_input = torch.empty(
            input_shape, dtype=self.input_dtype, device="cuda"
        )
        self.device_output = torch.empty(
            output_shape, dtype=self.output_dtype, device="cuda"
        )
        self.host_output = torch.empty(
            output_shape, dtype=self.output_dtype, device="cpu", pin_memory=True
        )
        self.context.set_tensor_address(self.input_name, self.device_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.device_output.data_ptr())
        self.input_shape = input_shape
        self.output_shape = output_shape

    def create_video_slots(
        self,
        input_shape: tuple[int, ...],
        frame_shape: tuple[int, int, int],
        slot_count: int,
        cuda_video_kernels: CudaVideoKernels,
    ) -> list[VideoPipelineSlot]:
        if slot_count < 2:
            raise ValueError("Video pipelining requires at least two host slots.")
        self.configure(input_shape)
        self.device_frame = torch.empty(frame_shape, dtype=torch.uint8, device="cuda")
        self.device_overlay = torch.empty(
            frame_shape, dtype=torch.uint8, device="cuda"
        )
        self.cuda_video_kernels = cuda_video_kernels

        slots: list[VideoPipelineSlot] = []
        for _ in range(slot_count):
            host_frame = torch.empty(
                frame_shape, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            host_overlay = torch.empty(
                frame_shape, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            slots.append(
                VideoPipelineSlot(
                    host_frame=host_frame,
                    host_frame_array=host_frame.numpy(),
                    host_overlay=host_overlay,
                    host_overlay_array=host_overlay.numpy(),
                )
            )
        return slots

    def process_video_slot(self, slot: VideoPipelineSlot, alpha: float) -> None:
        """Run GPU preprocessing, inference, and overlay for one decoded frame."""
        if (
            self.device_frame is None
            or self.device_overlay is None
            or self.device_input is None
            or self.device_output is None
            or self.cuda_video_kernels is None
        ):
            raise RuntimeError("Video resources have not been configured.")

        with torch.cuda.stream(self.inference_stream):
            self.device_frame.copy_(slot.host_frame, non_blocking=True)
            self.cuda_video_kernels.preprocess(
                self.device_frame, self.device_input, self.inference_stream
            )
            ok = self.context.execute_async_v3(
                stream_handle=self.inference_stream.cuda_stream
            )
            if not ok:
                raise RuntimeError("Failed to execute TensorRT engine.")
            self.cuda_video_kernels.overlay(
                self.device_frame,
                self.device_output,
                self.device_overlay,
                alpha,
                self.inference_stream,
            )
            slot.host_overlay.copy_(self.device_overlay, non_blocking=True)
        self.inference_stream.synchronize()

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference while reusing all context, stream, and buffer allocations."""
        self.configure(tuple(input_array.shape))
        assert self.host_input_array is not None
        assert self.host_input is not None
        assert self.device_input is not None
        assert self.device_output is not None
        assert self.host_output is not None

        if input_array is not self.host_input_array:
            np.copyto(self.host_input_array, input_array, casting="same_kind")
        with torch.cuda.stream(self.inference_stream):
            self.device_input.copy_(self.host_input, non_blocking=True)
            ok = self.context.execute_async_v3(
                stream_handle=self.inference_stream.cuda_stream
            )
            if not ok:
                raise RuntimeError("Failed to execute TensorRT engine.")
            self.host_output.copy_(self.device_output, non_blocking=True)
        self.inference_stream.synchronize()
        return self.host_output.float().numpy().copy()
