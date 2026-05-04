#!/usr/bin/env python3
"""Train SegFormer for binary drivable-area semantic segmentation."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoConfig,
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)


ID2LABEL = {0: "background", 1: "drivable"}
LABEL2ID = {"background": 0, "drivable": 1}
DEFAULT_IMAGE_SIZE = (1920, 1080)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SegFormer-B0 for binary drivable-area segmentation."
    )
    parser.add_argument(
        "--data-root",
        default="data/segformer_dataset",
        help="Prepared dataset root containing images/train, masks/train, and train.txt.",
    )
    parser.add_argument(
        "--model-name",
        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        help="Base Hugging Face SegFormer checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/segformer-drivable",
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=DEFAULT_IMAGE_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "Training size. Pass one value for square input or WIDTH HEIGHT. "
            "Defaults to 1920 1080."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=6e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional max training steps for smoke tests. Use -1 for full training.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=None,
        help="Optional exact Trainer logging step interval.",
    )
    parser.add_argument(
        "--logging-epochs",
        type=float,
        default=2.0,
        help="Trainer logging interval in epochs.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep.",
    )
    return parser.parse_args()


def read_split(split_path: Path) -> list[str]:
    with split_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


class DrivableDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        image_processor,
        image_size: tuple[int, int],
    ) -> None:
        self.root = root
        self.split = split
        self.image_processor = image_processor
        self.image_height, self.image_width = image_size
        self.image_ids = read_split(root / f"{split}.txt")
        self.image_dir = root / "images" / split
        self.mask_dir = root / "masks" / split

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_id = self.image_ids[index]
        image_path = self.image_dir / f"{image_id}.jpg"
        mask_path = self.mask_dir / f"{image_id}.png"

        image = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        image = cv.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=cv.INTER_LINEAR,
        )
        mask = cv.resize(
            mask,
            (self.image_width, self.image_height),
            interpolation=cv.INTER_NEAREST,
        )

        encoded = self.image_processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long(),
        }


def compute_logging_steps(dataset_size: int, args: argparse.Namespace) -> int:
    if args.logging_steps is not None:
        return max(1, args.logging_steps)

    steps_per_epoch = math.ceil(dataset_size / args.batch_size)
    return max(1, round(steps_per_epoch * args.logging_epochs))


def make_training_arguments(
    args: argparse.Namespace,
    train_dataset: DrivableDataset,
) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "eval_strategy": "no",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": compute_logging_steps(len(train_dataset), args),
        "disable_tqdm": True,
        "report_to": "none",
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
    }
    try:
        return TrainingArguments(**kwargs)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def read_image_size(override: Sequence[int]) -> tuple[int, int]:
    if len(override) == 1:
        size = int(override[0])
        return size, size
    if len(override) == 2:
        width, height = override
        return int(height), int(width)
    raise ValueError("--image-size expects one value or WIDTH HEIGHT.")


def build_image_processor(model_name: str, image_size: tuple[int, int]):
    height, width = image_size
    image_processor = AutoImageProcessor.from_pretrained(
        model_name,
        do_reduce_labels=False,
        do_resize=False,
    )
    image_processor.size = {"height": height, "width": width}
    image_processor.do_resize = False
    return image_processor


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    image_size = read_image_size(args.image_size)

    image_processor = build_image_processor(args.model_name, image_size)
    train_dataset = DrivableDataset(data_root, "train", image_processor, image_size)

    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(ID2LABEL)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=make_training_arguments(args, train_dataset),
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
