from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


DEFAULT_IMAGE_SIZE = (1920, 1080)


class SegformerLogitsWrapper(torch.nn.Module):
    def __init__(self, model: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Segformer model to ONNX format."
    )
    parser.add_argument(
        "--model-dir",
        default="models/segformer-drivable",
        help="Directory containing the trained Segformer model.",
    )
    parser.add_argument(
        "--output",
        default="models/segformer-drivable/segformer_drivable_1920x1080.onnx",
        help="Path to save the exported ONNX model.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=DEFAULT_IMAGE_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "Input size for the ONNX model. Pass one value for a square input "
            "or WIDTH HEIGHT. Defaults to 1920 1080."
        ),
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version to use for export.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device to use during export. Defaults to auto with CPU fallback.",
    )
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


def cuda_device_is_usable() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available in this PyTorch environment."

    accelerator_error = getattr(torch, "AcceleratorError", RuntimeError)
    try:
        probe = torch.randn(1, device="cuda")
        torch.cuda.synchronize()
        del probe
    except (RuntimeError, accelerator_error) as exc:
        return False, str(exc).splitlines()[0]

    return True, None


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")

    usable, reason = cuda_device_is_usable()
    if usable:
        return torch.device("cuda")

    if requested_device == "cuda":
        raise RuntimeError(
            "CUDA was requested but is not usable for ONNX export. "
            f"{reason} Try again with --device cpu."
        )

    print(
        f"Warning: CUDA is not usable for ONNX export ({reason}); "
        "falling back to CPU."
    )
    return torch.device("cpu")


def export_onnx(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    height, width = read_image_size(model_dir, args.image_size)

    device = resolve_device(args.device)
    if device.type != "cuda":
        print(
            "Exporting ONNX on CPU. The exported model can still be used "
            "to build a TensorRT engine."
        )

    model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    wrapped_model = SegformerLogitsWrapper(model)
    wrapped_model.to(device)

    dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=args.opset_version,
            do_constant_folding=True,
            dynamic_axes=None,
            export_params=True,
            dynamo=False,
        )

    print(f"ONNX model exported successfully to {output_path}")
    print(f"Model input shape: (1, 3, {height}, {width})")


def check_onnx(output_path: Path) -> None:
    import onnx

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("ONNX model is valid.")


def main() -> int:
    args = parse_args()
    export_onnx(args)
    if args.output:
        check_onnx(Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
