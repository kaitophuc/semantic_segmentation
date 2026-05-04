# Drivable-Area Semantic Segmentation

This project fine-tunes and runs a SegFormer-B0 model for binary drivable-area semantic segmentation.

The model predicts one mask class:

```text
0 = background / non-drivable
1 = drivable
```

## What Is Included

Included:

- Training, dataset preparation, inference, and ONNX export scripts.
- A final trained model in `models/segformer-drivable/`.
- Instructions for recreating the dataset from a CVAT export.

Not included:

- Raw camera images.
- Prepared training datasets.
- CVAT exports.
- GPS/KML/map files.
- Notebooks, tokens, virtual environments, checkpoints, and optimizer state.

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check CUDA/PyTorch:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## Recreate The Dataset

In CVAT, label only one semantic class:

```text
Drivable
```

Export the task or project as:

```text
Segmentation mask 1.1
```

The unzipped CVAT export should look like:

```text
cvat_export/
  JPEGImages/
  SegmentationClass/
  SegmentationObject/
  ImageSets/
    Segmentation/
      default.txt
  labelmap.txt
```

Convert it into class-ID PNG masks:

```bash
python prepare_dataset.py \
  --input-root cvat_export \
  --output-root data/segformer_dataset
```

Expected output:

```text
data/segformer_dataset/
  images/train/*.jpg
  images/val/*.jpg
  masks/train/*.png
  masks/val/*.png
  train.txt
  val.txt
  mapping.csv
```

The generated mask PNGs are single-channel images whose pixel values are only `0` and `1`.

## Training

Run a full training job:

```bash
python train.py \
  --data-root data/segformer_dataset \
  --output-dir models/segformer-drivable \
  --batch-size 1 \
  --epochs 30
```

Meaning of the main options:

- `--data-root`: prepared dataset root.
- `--image-size`: training input size. The default is `1920 1080`.
- `--batch-size`: lower this if CUDA runs out of memory.
- `--epochs`: number of training passes through the dataset.

For a quick smoke test:

```bash
python train.py --data-root data/segformer_dataset --max-steps 2 --epochs 1
```

If the machine has no internet access, use the included local model as the starting checkpoint:

```bash
python train.py \
  --data-root data/segformer_dataset \
  --model-name models/segformer-drivable \
  --max-steps 2 \
  --epochs 1
```

## Inference

Run TensorRT inference with the full-HD engine:

```bash
python inference.py \
  --image path/to/image.jpg \
  --engine models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine \
  --model-dir models/segformer-drivable \
  --output-mask outputs/mask.png \
  --overlay outputs/overlay.jpg
```

`outputs/mask.png` is a grayscale mask:

```text
0 = background / non-drivable
1 = drivable
```

The optional overlay colors drivable pixels green on top of the original image.

## Running On Another Machine

Clone the repository on the target machine, install dependencies, and run:

```bash
python inference.py \
  --image path/to/image.jpg \
  --engine models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine \
  --model-dir models/segformer-drivable \
  --output-mask outputs/mask.png
```

The Hugging Face model folder is portable. The TensorRT engine is not.

## TensorRT Deployment

Export ONNX:

```bash
python -m pip install onnx
python export_onnx.py \
  --model-dir models/segformer-drivable
```

Copy the ONNX model to the deployment machine and build the TensorRT engine there:

```bash
trtexec \
  --onnx=models/segformer-drivable/segformer_drivable_1920x1080.onnx \
  --saveEngine=models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine \
  --fp16
```

Build the `.engine` file on the exact inference machine whenever possible. TensorRT engines depend on GPU architecture, CUDA, TensorRT, driver/runtime compatibility, precision settings, and optimization profiles.

## Security And Data Policy

This public project intentionally excludes:

- Full datasets and raw camera recordings.
- GPS/KML map files and converted coordinate JSON.
- Notebooks that mention local token setup.
- Virtual environments and caches.
- Trainer checkpoints and optimizer state.

Only the final model weights and reusable code are meant to be public.

## Troubleshooting

CUDA out of memory:

- Use `--batch-size 1`.
- Use a smaller `--image-size` only when intentionally running a reduced-size test.

Wrong mask values:

- The training masks must contain only `0` and `1`.
- CVAT color masks must be converted with `prepare_dataset.py`.
- If your CVAT `Drivable` color differs, pass `--drivable-rgb R,G,B`.

`evaluation_strategy` or `eval_strategy` error:

- Different Transformers versions use different argument names.
- This training script tries `eval_strategy` first and falls back to `evaluation_strategy`.

TensorRT missing:

- TensorRT is not installed by `requirements.txt`.
- Install TensorRT using NVIDIA instructions for your CUDA/driver environment.

Image/mask path mismatch:

- Confirm every ID in `train.txt` and `val.txt` has a matching `.jpg` in `images/<split>/` and `.png` in `masks/<split>/`.
