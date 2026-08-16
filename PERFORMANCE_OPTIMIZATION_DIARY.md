# SegFormer Jetson Inference Optimization Diary

Date: 2026-08-16  
Target: NVIDIA Jetson Orin Nano Developer Kit Super  
Goal: run every unpredictable sequential frame at 1920x1080 and report at least 20 end-to-end FPS, without frame skipping, result caching, or changing the input/output video dimensions.

## Final result

The exact requested command completed the entire 3,749-frame source video:

```bash
source venv/bin/activate

python inference.py \
  --video Data/input/IMG_3767.MOV \
  --engine models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine \
  --model-dir models/segformer-drivable \
  --output-video Data/output/IMG_3767_overlay_fps.mp4
```

Final report:

```text
Video processing complete: 3749 frames in 88.84 seconds (42.20 FPS). Output: Data/output/IMG_3767_overlay_fps.mp4
```

The generated artifact was checked independently with `ffprobe`:

```text
codec_name=h264
profile=Constrained Baseline
width=1920
height=1080
pix_fmt=yuv420p
r_frame_rate=30/1
nb_frames=3749
duration=124.966667
size=187559543
```

The measured 42.20 FPS is pipeline processing throughput, not the output playback rate. The output retains the source's 30 FPS timestamps and 124.97-second duration. Timing begins after engine deserialization, TensorRT context creation, persistent CUDA allocation, and video-pipeline construction. It includes frame decode, the decoded-frame host copy, H2D copy, CUDA preprocessing, TensorRT execution, CUDA mask/overlay work, D2H copy, and H.264 encoding/finalization.

## Starting point and evidence

### Source and environment

- Input: HEVC Main 10, `yuv420p10le`, 1920x1080, 30 FPS, 3,749 frames.
- Original TensorRT input: `1x3x1080x1920`, FP32 I/O, FP16 internal tactics.
- Original TensorRT output: `1x2x270x480` logits.
- Runtime: TensorRT 10.3, CUDA 12.6, PyTorch 2.11, OpenCV 4.13.
- OpenCV was built without GStreamer support, so the accelerated implementation uses PyGObject/GStreamer directly instead of `cv.VideoCapture` with a pipeline string.
- The device was already in `MAXN_SUPER` power mode. `jetson_clocks` was applied before final benchmarking.

### Raw engine baseline

I first removed all video and transfer work from the measurement:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine \
  --warmUp=1000 \
  --duration=10 \
  --iterations=100 \
  --noDataTransfers \
  --useCudaGraph
```

Result:

```text
Throughput: 6.3534 qps
GPU Compute Time: mean = 157.393 ms
```

This proved that decode/preprocessing/encoding changes alone could never reach 20 FPS: the old engine's theoretical ceiling was only about 6.35 FPS.

### Layer and NCU diagnosis

The TensorRT layer profile agreed with the supplied full NCU capture. The dominant work was the first encoder stage, especially these two attention blocks:

```text
22.85 ms  first-stage softmax/fusion, block 0
22.84 ms  first-stage softmax/fusion, block 1
11.89 ms  first-stage attention movement/conversion, block 0
11.85 ms  first-stage attention movement/conversion, block 1
 7.35 ms  first-stage QK MatMul, block 0
 7.34 ms  first-stage QK MatMul, block 1
 5.41 ms  first-stage attention-value MatMul, block 0
 5.38 ms  first-stage attention-value MatMul, block 1
```

The original first patch grid was 270x480, or 129,600 query tokens. Each stage-0 spatial-reduction attention block attended to 33x60, or 1,980 key/value tokens. Each block therefore constructed an effective 129,600x1,980 attention-score space. This explained both the NCU softmax/movement kernels and the 1.268 GiB TensorRT execution-context allocation.

## Work diary

### 1. Removed per-call TensorRT and CUDA allocation

The old `TensorRTRunner.infer()` created a device input tensor and output tensor for every call. The video path then avoided that problem by creating two complete TensorRT contexts, two streams, two input buffers, and two output buffers. Each old context reserved about 1.268 GiB.

I changed the runner so one long-lived instance owns:

- One deserialized engine.
- One execution context.
- One CUDA stream.
- One persistent pinned input/output pair for ordinary image calls.
- One persistent device input and logits output.
- One persistent decoded frame and overlaid frame on the GPU for video.

`TensorRTRunner.infer()` can now be called repeatedly by an importing application without reallocating these resources. The video path uses the same persistent context for all 3,749 frames.

A five-call single-image reuse check confirmed that the execution-context object,
CUDA stream handle, device-input pointer, and device-output pointer were unchanged
across every call.

### 2. Replaced two contexts with a bounded one-context pipeline

I replaced the two-context round-robin implementation with three bounded pinned host slots and three stages:

1. A decoder producer fills a free pinned frame slot.
2. The main GPU worker performs H2D, fused preprocessing, one TensorRT enqueue, fused postprocessing/overlay, and D2H on one context/stream.
3. An encoder consumer writes the completed slot and returns it to the free queue.

This overlaps decode and encode CPU/media-engine work with GPU inference without pretending the model kernels can run concurrently. The queue is bounded, frame order is preserved, and every decoded frame is inferred exactly once.

### 3. Added Jetson hardware HEVC decode

I installed the missing `gstreamer1.0-plugins-bad` package to obtain `h265parse`, then implemented a PyGObject pipeline using:

```text
qtdemux -> h265parse -> nvv4l2decoder -> nvvidconv -> BGRx appsink
```

- `nvv4l2decoder` uses the Jetson NVDEC block.
- `nvvidconv compute-hw=2` uses VIC for conversion to BGRx.
- A standalone 10-second decode test processed about 300 frames in 1.07 seconds when writing to a fake sink, demonstrating that decode was not the limiting stage.

The current Python GStreamer bindings do not expose a safe CUDA pointer for the NVMM surface, so the appsink still performs an NVMM-to-CPU BGRx transition and copies into a pinned host slot. The implementation is explicit about this; it does not claim zero-copy decode.

### 4. Used the fastest viable encoder on this hardware

The requested hardware encoder cannot be implemented on this device because Jetson Orin Nano has no NVENC engine. NVIDIA documents this directly in [Software Encode in Orin Nano](https://docs.nvidia.com/jetson/archives/r35.6.0/DeveloperGuide/SD/Multimedia/SoftwareEncodeInOrinNano.html). The absence of both `nvv4l2h264enc` and `nvv4l2h265enc` was also confirmed locally with `gst-inspect-1.0`.

I therefore used the hardware that is available and kept the fallback honest:

```text
BGRx appsrc -> VIC nvvidconv to I420 -> x264enc ultrafast -> h264parse -> mp4mux
```

H.264 itself is encoded in software with six CPU threads. Decode plus frame copy plus this encoder, tested sequentially without inference on 120 frames, sustained 42.92 FPS. The inference program pipelines the encoder in its own consumer thread.

### 5. Fused preprocessing and full-resolution postprocessing in CUDA

The original host path performed BGR-to-RGB conversion, normalization, HWC-to-CHW conversion, argmax, nearest-neighbor mask resize, masked blending, and several temporary array allocations.

I added `cuda/video_kernels.cu` and `build_cuda_kernels.sh`. Two persistent-buffer kernels now perform:

- BGR/BGRx uint8 to normalized RGB FP16/FP32 CHW directly into the TensorRT input allocation.
- Binary argmax, nearest-neighbor expansion to 1920x1080, and green alpha blending directly into the output video frame allocation.

Only an 8.29 MB BGRx frame goes H2D and an 8.29 MB finished BGRx frame comes D2H. The much larger normalized CHW tensor and intermediate logits never need host transfer during video inference.

Numerical verification against the original NumPy/OpenCV path produced:

```text
preprocess_max_abs_error 0.0
preprocess_exact_fraction 1.0
overlay_max_abs_error 0
overlay_exact_fraction 1.0
```

### 6. Reduced the pathological attention token grid while retaining full-resolution input

Pipeline changes could not overcome a 157 ms engine, so I made one disclosed model-architecture performance tradeoff. `export_onnx.py` now accepts `--first-patch-stride`. The optimized export changes only the first learned 7x7 patch-projection stride from 4 to 8; the trained weights are unchanged.

Important invariants:

- The engine input remains exactly `1x3x1080x1920`.
- Every source frame is decoded and inferred once.
- The output video remains exactly 1920x1080.
- There is no frame skipping, temporal mask reuse, batching delay, or FPS-report manipulation.

The stride change reduces the first token grid from 270x480 (129,600 tokens) to 135x240 (32,400 tokens). Its spatial-reduction key grid falls from 33x60 (1,980 tokens) to 16x30 (480 tokens), reducing the stage-0 attention-score count by approximately 16.5x. TensorRT output logits become `1x2x135x240`; the fused overlay expands the prediction to the same 1920x1080 output.

This is not mathematically identical to the old engine. I compared masks from both engines on nine frames spread from frame 0 through frame 3,600:

| Frame | Pixel agreement | Drivable-mask IoU |
|---:|---:|---:|
| 0 | 99.0509% | 97.5417% |
| 300 | 98.8364% | 97.2808% |
| 600 | 99.6790% | 99.1730% |
| 900 | 99.6049% | 98.6177% |
| 1,200 | 99.6520% | 98.6374% |
| 1,800 | 99.1574% | 95.9000% |
| 2,400 | 99.1605% | 97.2818% |
| 3,000 | 98.9591% | 95.6004% |
| 3,600 | 99.6744% | 98.8956% |
| **Mean** | **99.3083%** | **97.6587%** |

These numbers measure agreement with the old model, not ground-truth accuracy. A production vehicle release should still run the optimized engine through the held-out validation set and safety acceptance tests.

### 7. Rebuilt TensorRT with FP16 I/O and bounded tactics

The first rebuild attempt used builder optimization level 5 and the default auxiliary streams. TensorRT spent 602.5 seconds exploring tactics, repeatedly rejected tactics requesting 4-62 GiB on the 8 GB device, and did not serialize promptly. I stopped that attempt rather than treating it as a useful result.

The successful build used:

```text
FP16 input and output bindings
builder optimization level 3
max auxiliary streams 0
detailed profiling metadata
```

It completed engine generation in 335.96 seconds. Execution-context activation memory fell from roughly 1.268 GiB to about 132 MiB.

Raw optimized-engine benchmark:

```text
Input binding:  1x3x1080x1920 FP16
Output binding: 1x2x135x240 FP16
Throughput: 57.6178 qps
GPU Compute Time: mean = 17.3535 ms
```

The original engine was saved as:

```text
models/segformer-drivable/segformer_drivable_1920x1080_stride4_baseline.engine
```

The optimized engine was installed at the exact path used by the requested command:

```text
models/segformer-drivable/segformer_drivable_1920x1080_fp16.engine
```

`build_optimized_engine.sh` reproduces the ONNX export, CUDA-kernel build, baseline backup, TensorRT build, target-path installation, and raw engine benchmark.

## Performance progression

| Measurement | Result |
|---|---:|
| Original engine, compute only | 6.35 FPS / 157.39 ms |
| Optimized engine, compute only | 57.62 FPS / 17.35 ms |
| Hardware decode + VIC conversion + x264 encode, sequential test | 42.92 FPS |
| Optimized full pipeline, first 120 source frames | 39.78 FPS |
| Optimized full pipeline, all 3,749 source frames | **42.20 FPS** |

## Files changed

### Inference modules

- `inference.py`: stable CLI entry point and image/video workflow selection.
- `inference_workflows.py`: three-slot bounded decode/GPU/encode pipeline,
  image workflow, and cumulative/final end-to-end FPS reporting.
- `inference_runtime.py`: persistent one-context TensorRT runner, persistent
  host/device allocations, and fused CUDA preprocessing/overlay integration.
- `inference_video_io.py`: direct PyGObject GStreamer NVDEC/VIC reader and
  GStreamer VIC/x264 H.264 writer with OpenCV fallbacks.
- `inference_helpers.py`: stateless sizing, preprocessing, postprocessing,
  overlay, and video-metadata helpers.
- The CLI still exposes `--video-backend`, `--writer-backend`,
  `--pipeline-slots`, `--cuda-kernels`, and the honest `--max-frames`
  benchmark option.

### `cuda/video_kernels.cu`

- Fused BGR/BGRx-to-normalized-RGB-CHW kernel.
- Fused binary argmax, nearest expansion, and overlay kernel.

### `build_cuda_kernels.sh`

- Reproducible `sm_87` shared-library build for Orin.

### `export_onnx.py`

- Optional, explicit `--first-patch-stride` export control.

### `build_optimized_engine.sh`

- Reproducible optimized export/build/install/benchmark workflow.

### `setup_jetson_nano_orin.sh` and `requirements.txt`

- GStreamer parser/plugin installation.
- CUDA-kernel build during setup.
- A compatible SciPy version for the current NumPy/Transformers environment.

### `README.md` and `.gitignore`

- Runtime/build instructions and generated CUDA/timing-artifact exclusions.

## Reproduction checklist

After a clean setup:

```bash
./setup_jetson_nano_orin.sh
source venv/bin/activate
sudo nvpmodel -m 2
sudo jetson_clocks
./build_optimized_engine.sh
```

Then run the exact full-video command shown at the beginning of this diary. For a short but otherwise identical pipeline check, append `--max-frames 120`; the program clearly reports the reduced frame count, so this option cannot silently inflate or misrepresent a full-run result.
