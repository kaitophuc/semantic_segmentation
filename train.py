#!/usr/bin/env python3
"""Train SegFormer for binary drivable-area semantic segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)


ID2LABEL = {0: "background", 1: "drivable"}
LABEL2ID = {"background": 0, "drivable": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SegFormer-B0 for binary drivable-area segmentation."
    )
    parser.add_argument(
        "--data-root",
        default="data/segformer_dataset",
        help="Prepared dataset root containing images, masks, train.txt, and val.txt.",
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
        default=1024,
        help="Square training size in pixels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train/eval batch size.",
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
        default=10,
        help="Trainer logging interval.",
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
        image_size: int,
    ) -> None:
        self.root = root
        self.split = split
        self.image_processor = image_processor
        self.image_size = image_size
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
            (self.image_size, self.image_size),
            interpolation=cv.INTER_LINEAR,
        )
        mask = cv.resize(
            mask,
            (self.image_size, self.image_size),
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


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    logits_tensor = torch.as_tensor(logits)
    labels_tensor = torch.as_tensor(labels)

    logits_tensor = torch.nn.functional.interpolate(
        logits_tensor,
        size=labels_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    predictions = logits_tensor.argmax(dim=1)

    pixel_accuracy = (predictions == labels_tensor).float().mean().item()

    pred_drivable = predictions == LABEL2ID["drivable"]
    label_drivable = labels_tensor == LABEL2ID["drivable"]
    intersection = torch.logical_and(pred_drivable, label_drivable).sum().item()
    union = torch.logical_or(pred_drivable, label_drivable).sum().item()
    drivable_iou = intersection / union if union > 0 else 0.0

    return {
        "pixel_accuracy": pixel_accuracy,
        "drivable_iou": drivable_iou,
    }


def make_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "drivable_iou",
        "greater_is_better": True,
    }
    try:
        return TrainingArguments(**kwargs)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def build_image_processor(model_name: str, image_size: int):
    image_processor = AutoImageProcessor.from_pretrained(
        model_name,
        do_reduce_labels=False,
        do_resize=False,
    )
    image_processor.size = {"height": image_size, "width": image_size}
    image_processor.do_resize = False
    return image_processor


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)

    image_processor = build_image_processor(args.model_name, args.image_size)
    train_dataset = DrivableDataset(data_root, "train", image_processor, args.image_size)
    val_dataset = DrivableDataset(data_root, "val", image_processor, args.image_size)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=make_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

