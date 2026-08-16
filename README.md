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

On Jetson Orin Nano, the project setup script also installs the GStreamer
elements used by hardware HEVC decode and builds the fused CUDA video kernels:

```bash
./setup_jetson_nano_orin.sh
source venv/bin/activate
```

Build and install the performance-optimized 1920x1080 TensorRT engine:

```bash
sudo nvpmodel -m 2
sudo jetson_clocks
./build_optimized_engine.sh
```

The build script preserves the original stride-4 engine as
`segformer_drivable_1920x1080_stride4_baseline.engine`, exports the optimized
model, and installs the optimized engine at the path used by the inference
commands below.

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
  masks/train/*.png
  train.txt
  mapping.csv
```

The generated mask PNGs are single-channel images whose pixel values are only `0` and `1`.

## Training

Run a full training job:

```bash
python train.py \
  --data-root data/segformer_dataset \
  --output-dir models/segformer-drivable \
  --batch-size 2 \
  --epochs 30
```

Meaning of the main options:

- `--data-root`: prepared dataset root.
- `--image-size`: training input size. The default is `1920 1080`.
- `--batch-size`: lower this if CUDA runs out of memory.
- `--epochs`: number of training passes through the dataset.

Training uses every sample in `train.txt`; no validation split is required by default.

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

The inference code is split by responsibility so the main path can be read
without stepping through backend details:

- `inference.py` parses the CLI and selects the image or video workflow.
- `inference_workflows.py` contains the important end-to-end control flow.
- `inference_runtime.py` owns the persistent TensorRT context, CUDA stream, and
  buffers.
- `inference_video_io.py` contains the GStreamer and OpenCV decoder/encoder
  implementations.
- `inference_helpers.py` groups stateless sizing, preprocessing,
  postprocessing, overlay, and video-metadata helpers.

For a code review, start with `inference.py`, continue into
`inference_workflows.py`, and then open the runtime or video-I/O module whose
implementation you need to inspect.

Run TensorRT inference on a video and write the segmentation overlay:

```bash
python inference.py \
  --video Data/input/IMG_3767.MOV \
  --engine models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine \
  --model-dir models/segformer-drivable \
  --output-video Data/output/IMG_3767_overlay_fps.mp4
```

On the tested Jetson Orin Nano Super, the complete 3,749-frame input produced:

```text
Video processing complete: 3749 frames in 88.84 seconds (42.20 FPS). Output: Data/output/IMG_3767_overlay_fps.mp4
```

Processing FPS measures the complete video pipeline: frame decoding,
preprocessing, TensorRT execution, mask and overlay generation, and output video
encoding. It starts only after the model, video streams, TensorRT contexts, and
GPU buffers are initialized, so model loading and other setup work are excluded.
The default Jetson path uses NVDEC hardware HEVC decoding, VIC color conversion,
fused CUDA preprocessing/overlay kernels, one persistent TensorRT context, and a
software x264 encoder. Orin Nano has no NVENC hardware encoder.

For a short, clearly labeled pipeline benchmark, append:

```bash
--max-frames 120
```

See [PERFORMANCE_OPTIMIZATION_DIARY.md](PERFORMANCE_OPTIMIZATION_DIARY.md) for
the baseline profile, failed experiments, model tradeoff, mask-agreement check,
and exact end-to-end verification.

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

For a continuously running application that submits individual camera frames,
import `load_engine` and `TensorRTRunner`, construct the runner once, and reuse
that runner's `infer()` method. The runner now retains its TensorRT execution
context, CUDA stream, pinned host buffers, and device buffers across calls;
starting a fresh `inference.py` process for each frame would still pay model and
CUDA cold-start costs.

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

For the original stride-4 architecture, copy the ONNX model to the deployment
machine and build the TensorRT engine there:

```bash
trtexec \
  --onnx=models/segformer-drivable/segformer_drivable_1920x1080.onnx \
  --saveEngine=models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine \
  --fp16
```

That original architecture measured only 6.35 compute-only FPS at 1920x1080 on
the tested Orin Nano. To reproduce the optimized full-resolution-input engine
used for the 42.20 FPS result, run `./build_optimized_engine.sh` instead. The
optimized export changes the first learned patch-projection stride from 4 to 8;
the input remains `1x3x1080x1920`, every frame is inferred once, and the output
video remains 1920x1080. This tradeoff achieved 99.31% mean pixel agreement and
97.66% mean drivable-mask IoU against the old engine on nine sampled drive
frames, but it still requires validation against ground-truth data before a
safety-critical release.

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

- Confirm every ID in `train.txt` has a matching `.jpg` in `images/train/` and `.png` in `masks/train/`.
