"""Command-line entry point for TensorRT SegFormer inference.

Start here to understand the program. The execution details live in:

* inference_workflows.py: single-image and full-video control flow
* inference_runtime.py: persistent TensorRT/CUDA resources
* inference_video_io.py: decoder and encoder implementations
* inference_helpers.py: stateless preprocessing and configuration helpers
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Import the runtime first. On JetPack, TensorRT must initialize its shared
# libraries before PyTorch is loaded.
from inference_runtime import TensorRTRunner, load_engine
from inference_helpers import (
    DEFAULT_CUDA_KERNELS,
    DEFAULT_INPUT_VIDEO,
    DEFAULT_OUTPUT_VIDEO,
    check_cuda_ready,
    choose_input_size,
)
from inference_workflows import run_image, run_video


# Keep the documented application-import surface available after the refactor.
__all__ = ["TensorRTRunner", "load_engine", "main", "run_image", "run_video"]


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
        help=(
            "Overlay output video path. "
            "Defaults to Data/output/IMG_3767_overlay.mp4."
        ),
    )
    parser.add_argument("--image", default=None, help="Optional single image path.")
    parser.add_argument(
        "--output-mask", default=None, help="Optional single image mask path."
    )
    parser.add_argument(
        "--overlay", default=None, help="Optional single image overlay path."
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity.")
    parser.add_argument(
        "--video-backend",
        choices=("auto", "gstreamer", "opencv"),
        default="auto",
        help="Video decoder backend. Auto prefers Jetson GStreamer hardware decode.",
    )
    parser.add_argument(
        "--writer-backend",
        choices=("auto", "gstreamer", "opencv"),
        default="auto",
        help="Video encoder backend. Auto prefers the pipelined GStreamer writer.",
    )
    parser.add_argument(
        "--pipeline-slots",
        type=int,
        default=3,
        help="Bounded host-frame slots shared by decode, GPU inference, and encode.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for honest short-run benchmarking.",
    )
    parser.add_argument(
        "--cuda-kernels",
        default=str(DEFAULT_CUDA_KERNELS),
        help="Fused CUDA preprocessing/overlay library built by build_cuda_kernels.sh.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)

    check_cuda_ready()
    engine = load_engine(Path(args.engine))
    runner = TensorRTRunner(engine)
    input_size = choose_input_size(
        engine,
        runner.input_name,
        model_dir,
        args.image_size,
    )

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
            video_backend=args.video_backend,
            writer_backend=args.writer_backend,
            pipeline_slots=args.pipeline_slots,
            max_frames=args.max_frames,
            cuda_kernels_path=Path(args.cuda_kernels),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
