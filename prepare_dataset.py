#!/usr/bin/env python3
"""Prepare CVAT segmentation masks for binary drivable-area training."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected RGB value like 71,183,114")

    try:
        rgb = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB values must be integers") from exc

    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("RGB values must be in range 0..255")
    return rgb  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a CVAT 'Segmentation mask 1.1' export into numbered "
            "image/mask pairs with grayscale class-ID masks."
        )
    )
    parser.add_argument(
        "--input-root",
        default="cvat_export",
        help=(
            "CVAT/VOC export root containing JPEGImages, SegmentationClass, "
            "and ImageSets/Segmentation/default.txt."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/segformer_dataset",
        help="Output folder for the prepared dataset.",
    )
    parser.add_argument(
        "--split-file",
        default="ImageSets/Segmentation/default.txt",
        help="Image ID list relative to input-root.",
    )
    parser.add_argument(
        "--drivable-rgb",
        type=parse_rgb,
        default=(71, 183, 114),
        help="RGB color used by CVAT for the Drivable label.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Optional validation split ratio, between 0 and 1. Defaults to train-only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/validation split.",
    )
    parser.add_argument(
        "--prefix",
        default="img_",
        help="Prefix for numbered output filenames.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Zero-padding width for numbered filenames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output folder.",
    )
    return parser.parse_args()


def read_ids(split_path: Path) -> list[str]:
    ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
    return [image_id for image_id in ids if image_id and not image_id.startswith("#")]


def make_split(ids: list[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be >= 0 and < 1")

    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    val_ids = set(shuffled[:val_count])
    train_ids = set(shuffled[val_count:])
    return train_ids, val_ids


def ensure_output_dirs(output_root: Path, include_val: bool) -> None:
    subsets = ("train", "val") if include_val else ("train",)
    for subset in subsets:
        (output_root / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_root / "masks" / subset).mkdir(parents=True, exist_ok=True)


def convert_mask(mask_path: Path, drivable_rgb: tuple[int, int, int]) -> np.ndarray:
    mask_bgr = cv.imread(str(mask_path), cv.IMREAD_COLOR)
    if mask_bgr is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    drivable_bgr = np.array(
        [drivable_rgb[2], drivable_rgb[1], drivable_rgb[0]],
        dtype=np.uint8,
    )
    drivable_pixels = np.all(mask_bgr == drivable_bgr, axis=2)

    class_id_mask = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    class_id_mask[drivable_pixels] = 1

    unknown_pixels = np.logical_and(
        np.any(mask_bgr != 0, axis=2),
        ~drivable_pixels,
    )
    if np.any(unknown_pixels):
        colors = np.unique(mask_bgr[unknown_pixels].reshape(-1, 3), axis=0)
        raise ValueError(
            f"Unknown non-background colors in {mask_path}: "
            f"{colors[:10].tolist()}"
        )

    return class_id_mask


def write_list(path: Path, stems: list[str]) -> None:
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    split_path = input_root / args.split_file

    if args.digits < 1:
        print("Error: --digits must be >= 1.")
        return 1
    if not split_path.exists():
        print(f"Error: split file does not exist: {split_path}")
        return 1
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        print(f"Error: output folder is not empty: {output_root}")
        print("Pass --overwrite to write into it anyway.")
        return 1

    image_ids = read_ids(split_path)
    if not image_ids:
        print(f"Error: no image IDs found in: {split_path}")
        return 1

    train_ids, val_ids = make_split(image_ids, args.val_ratio, args.seed)
    ensure_output_dirs(output_root, include_val=bool(val_ids))

    mapping_path = output_root / "mapping.csv"
    train_stems: list[str] = []
    val_stems: list[str] = []
    copied = 0

    with mapping_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["subset", "new_stem", "source_image", "source_mask"])

        for index, source_stem in enumerate(image_ids, start=1):
            subset = "val" if source_stem in val_ids else "train"
            new_stem = f"{args.prefix}{index:0{args.digits}d}"

            source_image = input_root / "JPEGImages" / f"{source_stem}.jpg"
            source_mask = input_root / "SegmentationClass" / f"{source_stem}.png"
            if not source_image.exists():
                print(f"Error: missing image: {source_image}")
                return 1
            if not source_mask.exists():
                print(f"Error: missing mask: {source_mask}")
                return 1

            image = cv.imread(str(source_image), cv.IMREAD_COLOR)
            if image is None:
                print(f"Error: could not read image: {source_image}")
                return 1

            class_id_mask = convert_mask(source_mask, args.drivable_rgb)
            if image.shape[:2] != class_id_mask.shape:
                print(
                    "Error: image/mask size mismatch: "
                    f"{source_image} {image.shape[:2]} vs "
                    f"{source_mask} {class_id_mask.shape}"
                )
                return 1

            output_image = output_root / "images" / subset / f"{new_stem}.jpg"
            output_mask = output_root / "masks" / subset / f"{new_stem}.png"
            shutil.copy2(source_image, output_image)
            if not cv.imwrite(str(output_mask), class_id_mask):
                print(f"Error: failed to write mask: {output_mask}")
                return 1

            if subset == "train":
                train_stems.append(new_stem)
            else:
                val_stems.append(new_stem)
            writer.writerow([subset, new_stem, str(source_image), str(source_mask)])
            copied += 1

    write_list(output_root / "train.txt", train_stems)
    if val_stems:
        write_list(output_root / "val.txt", val_stems)

    print(f"Done. Converted {copied} image/mask pairs.")
    print(f"Train: {len(train_stems)}")
    if val_stems:
        print(f"Val: {len(val_stems)}")
    print(f"Output: {output_root}")
    print(f"Mapping: {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
