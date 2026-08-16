from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys

SYSTEM_TENSORRT_PATH = Path("/usr/lib/python3.10/dist-packages")
if SYSTEM_TENSORRT_PATH.exists() and str(SYSTEM_TENSORRT_PATH) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TENSORRT_PATH))

import tensorrt as trt

if sys.path[0] == str(SYSTEM_TENSORRT_PATH):
    sys.path.pop(0)

import cv2 as cv
import numpy as np
import torch


LOGGER = trt.Logger(trt.Logger.WARNING)
DEFAULT_INPUT_VIDEO = Path("Data/input/IMG_3767.MOV")
DEFAULT_OUTPUT_VIDEO = Path("Data/output/IMG_3767_overlay.mp4")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SCALE = (1.0 / (255.0 * IMAGENET_STD)).astype(np.float32)
INPUT_BIAS = (-IMAGENET_MEAN / IMAGENET_STD).astype(np.float32)
BGR_TO_RGB_CHANNELS = (2, 1, 0)
PIXEL_VALUES = np.arange(256, dtype=np.float32)[:, None]
INPUT_LOOKUP_FLOAT16 = (
    (PIXEL_VALUES / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
).T.astype(np.float16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TensorRT SegFormer inference on an image or full video."
    )
    parser.add_argument("--engine", required=True, help="TensorRT engine path.")
    parser.add_argument(
        "--model-dir",
        default="models/segformer-drivable",
        help="Hugging Face model directory with preprocessor_config.json.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Engine input size. Pass one value for square input or WIDTH HEIGHT.",
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_INPUT_VIDEO),
        help="Input video path. Defaults to Data/input/IMG_3767.MOV.",
    )
    parser.add_argument(
        "--output-video",
        default=str(DEFAULT_OUTPUT_VIDEO),
        help="Overlay output video path. Defaults to Data/output/IMG_3767_overlay.mp4.",
    )
    parser.add_argument("--image", default=None, help="Optional single image path.")
    parser.add_argument("--output-mask", default=None, help="Optional single image mask path.")
    parser.add_argument("--overlay", default=None, help="Optional single image overlay path.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity.")
    return parser.parse_args()


def parse_image_size(override: Sequence[int]) -> tuple[int, int]:
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


def load_engine(engine_path: Path) -> trt.ICudaEngine:
    runtime = trt.Runtime(LOGGER)
    with engine_path.open("rb") as file:
        engine = runtime.deserialize_cuda_engine(file.read())

    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")
    return engine


def check_cuda_ready() -> None:
    if torch.cuda.is_available():
        return

    raise RuntimeError(
        "CUDA is not available to PyTorch, so TensorRT cannot run either. "
        "On Jetson, make sure PyTorch/TensorRT come from the JetPack-compatible "
        "NVIDIA packages and that the GPU device nodes are available."
    )


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
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


def get_io_names(engine: trt.ICudaEngine) -> tuple[str, str]:
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


@dataclass
class AsyncInferenceSlot:
    context: trt.IExecutionContext
    stream: torch.cuda.Stream
    done_event: torch.cuda.Event
    host_input: torch.Tensor
    host_input_array: np.ndarray
    device_input: torch.Tensor
    device_output: torch.Tensor
    host_output: torch.Tensor
    frame_bgr: np.ndarray | None = None
    busy: bool = False

class TensorRTRunner:
    def __init__(self, engine: trt.ICudaEngine) -> None:
        self.engine = engine
        self.context = engine.create_execution_context()
        self.input_name, self.output_name = get_io_names(engine)
        self.input_dtype = torch_dtype_from_trt(engine.get_tensor_dtype(self.input_name))
        self.output_dtype = torch_dtype_from_trt(engine.get_tensor_dtype(self.output_name))
        self.inference_stream = torch.cuda.Stream()

    def _configure_context(self, context: trt.IExecutionContext, input_shape: tuple[int, ...]) -> tuple[int, ...]:
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
    
    def create_async_slots(self, input_shape: tuple[int, ...], slot_count: int = 2) -> list[AsyncInferenceSlot]:

        slots = []
        for _ in range(slot_count):
            context = self.engine.create_execution_context()
            output_shape = self._configure_context(context, input_shape)

            host_input = torch.empty(input_shape, dtype=self.input_dtype, device="cpu", pin_memory=True)
            device_input = torch.empty(input_shape, dtype=self.input_dtype, device="cuda")
            device_output = torch.empty(output_shape, dtype=self.output_dtype, device="cuda")
            host_output = torch.empty(output_shape, dtype=self.output_dtype, device="cpu", pin_memory=True)

            slots.append(
                AsyncInferenceSlot(
                    context=context,
                    stream=torch.cuda.Stream(),
                    done_event=torch.cuda.Event(blocking=False),
                    host_input=host_input,
                    host_input_array=host_input.numpy(),
                    device_input=device_input,
                    device_output=device_output,
                    host_output=host_output,
                )
            )

        return slots
    
    def enqueue_async(self, slot: AsyncInferenceSlot, frame_bgr: np.ndarray) -> None:
        if slot.busy:
            raise RuntimeError("Tried to enqueue into a busy async inference slot.")

        slot.frame_bgr = frame_bgr

        with torch.cuda.stream(slot.stream):
            slot.device_input.copy_(slot.host_input, non_blocking=True)

            slot.context.set_tensor_address(self.input_name, slot.device_input.data_ptr())
            slot.context.set_tensor_address(self.output_name, slot.device_output.data_ptr())

            ok = slot.context.execute_async_v3(stream_handle=slot.stream.cuda_stream)
            if not ok:
                raise RuntimeError("Failed to execute TensorRT engine.")
            
            slot.host_output.copy_(slot.device_output, non_blocking=True)
            slot.done_event.record(slot.stream)

        slot.busy = True

    def finish_async(self, slot: AsyncInferenceSlot) -> tuple[np.ndarray, np.ndarray]:
        if not slot.busy:
            raise RuntimeError("Tried to finish an idle async inference slot.")
        
        slot.done_event.synchronize()

        if slot.frame_bgr is None:
            raise RuntimeError("Async slot finished without an attached frame.")
        
        frame_bgr = slot.frame_bgr
        logits = slot.host_output.numpy()

        slot.frame_bgr = None
        slot.busy = False

        return frame_bgr, logits
    
    def infer(self, input_array: np.ndarray) -> np.ndarray:
        with torch.cuda.stream(self.inference_stream):
            input_tensor = torch.from_numpy(input_array).to("cuda").to(self.input_dtype)
            input_shape = tuple(input_tensor.shape)
            output_shape = self._configure_context(self.context, input_shape)

            output_tensor = torch.empty(output_shape, dtype=self.output_dtype, device="cuda")

            self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
            self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())

            ok = self.context.execute_async_v3(
                stream_handle=self.inference_stream.cuda_stream
            )
            if not ok:
                raise RuntimeError("Failed to execute TensorRT engine.")
        
        self.inference_stream.synchronize()
        return output_tensor.detach().cpu().float().numpy()


def postprocess(logits: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
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


def run_image(
    runner: TensorRTRunner,
    image_path: Path,
    input_size: tuple[int, int],
    output_mask_path: Path | None,
    overlay_path: Path | None,
    alpha: float,
) -> None:
    frame = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    logits = runner.infer(preprocess_frame(frame, input_size))
    mask = postprocess(logits, frame.shape[:2])

    if output_mask_path is not None:
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(output_mask_path), mask)

    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(overlay_path), make_overlay(frame, mask, alpha))


def run_video(
    runner: TensorRTRunner,
    video_path: Path,
    output_path: Path,
    input_size: tuple[int, int],
    alpha: float,
) -> None:
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video not found or could not be opened: {video_path}")

    fps = capture.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_w = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_h = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    input_h, input_w = input_size
    input_shape = (1, 3, input_h, input_w)
    slots = runner.create_async_slots(input_shape=input_shape, slot_count=2)

    scheduled_frames = 0
    completed_frames = 0
    next_slot_index = 0

    def finish_and_write(slot: AsyncInferenceSlot) -> None:
        nonlocal completed_frames

        frame, logits = runner.finish_async(slot)
        mask = postprocess(logits, frame.shape[:2])
        writer.write(make_overlay(frame, mask, alpha))

        completed_frames += 1
        if completed_frames % 30 == 0:
            print(f"Processed {completed_frames}/{total_frames or '?'} frames")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            slot = slots[next_slot_index]

            if slot.busy:
                finish_and_write(slot)

            preprocess_frame(frame, input_size, output=slot.host_input_array)
            runner.enqueue_async(slot, frame)

            scheduled_frames += 1
            next_slot_index = (next_slot_index + 1) % len(slots)
        
        for slot in slots:
            if slot.busy:
                finish_and_write(slot)

    finally:
        capture.release()
        writer.release()

    print(f"Wrote overlay video: {output_path}")

def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    check_cuda_ready()
    engine = load_engine(Path(args.engine))
    runner = TensorRTRunner(engine)
    input_size = choose_input_size(engine, runner.input_name, model_dir, args.image_size)

    if args.image:
        run_image(
            runner=runner,
            image_path=Path(args.image),
            input_size=input_size,
            output_mask_path=Path(args.output_mask) if args.output_mask else None,
            overlay_path=Path(args.overlay) if args.overlay else None,
            alpha=args.alpha,
        )
    else:
        run_video(
            runner=runner,
            video_path=Path(args.video),
            output_path=Path(args.output_video),
            input_size=input_size,
            alpha=args.alpha,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
