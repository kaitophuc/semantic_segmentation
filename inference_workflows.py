"""High-level single-image and pipelined video inference workflows."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import time

import cv2 as cv

from inference_helpers import (
    VideoInfo,
    make_overlay,
    postprocess,
    preprocess_frame,
    probe_video,
)
from inference_runtime import CudaVideoKernels, TensorRTRunner, VideoPipelineSlot
from inference_video_io import (
    VideoReader,
    VideoWriter,
    create_video_reader,
    create_video_writer,
)


def run_image(
    runner: TensorRTRunner,
    image_path: Path,
    input_size: tuple[int, int],
    output_mask_path: Path | None,
    overlay_path: Path | None,
    alpha: float,
) -> None:
    """Run the readable CPU pre/inference/post path for one image."""
    frame = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    input_h, input_w = input_size
    runner.configure((1, 3, input_h, input_w))
    assert runner.host_input_array is not None
    preprocess_frame(frame, input_size, output=runner.host_input_array)
    logits = runner.infer(runner.host_input_array)
    mask = postprocess(logits, frame.shape[:2])

    if output_mask_path is not None:
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(output_mask_path), mask)

    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(overlay_path), make_overlay(frame, mask, alpha))


class VideoPipeline:
    """Coordinate decode, one-context GPU processing, and encode workers."""

    def __init__(
        self,
        runner: TensorRTRunner,
        reader: VideoReader,
        writer: VideoWriter,
        slots: list[VideoPipelineSlot],
        info: VideoInfo,
        output_path: Path,
        alpha: float,
        max_frames: int | None,
    ) -> None:
        self.runner = runner
        self.reader = reader
        self.writer = writer
        self.info = info
        self.output_path = output_path
        self.alpha = alpha
        self.max_frames = max_frames

        self.free_slots: queue.Queue[VideoPipelineSlot] = queue.Queue()
        self.decoded_slots: queue.Queue[VideoPipelineSlot | None] = queue.Queue()
        self.completed_slots: queue.Queue[VideoPipelineSlot | None] = queue.Queue()
        self.errors: queue.Queue[BaseException] = queue.Queue()
        self.stop_event = threading.Event()
        for slot in slots:
            self.free_slots.put(slot)

        self.completed_frames = 0
        total_to_process = info.total_frames
        if max_frames is not None:
            total_to_process = (
                min(total_to_process, max_frames)
                if total_to_process > 0
                else max_frames
            )
        self.total_frames_display: int | str = (
            total_to_process if total_to_process > 0 else "?"
        )
        self.processing_started_at = 0.0

    def _decode_worker(self) -> None:
        frame_index = 0
        try:
            while not self.stop_event.is_set():
                if self.max_frames is not None and frame_index >= self.max_frames:
                    break
                try:
                    slot = self.free_slots.get(timeout=0.1)
                except queue.Empty:
                    continue
                if not self.reader.read_into(slot.host_frame_array):
                    self.free_slots.put(slot)
                    break
                slot.frame_index = frame_index
                self.decoded_slots.put(slot)
                frame_index += 1
        except BaseException as exc:
            self.errors.put(exc)
            self.stop_event.set()
        finally:
            try:
                self.reader.close()
            except BaseException as exc:
                self.errors.put(exc)
                self.stop_event.set()
            finally:
                self.decoded_slots.put(None)

    def _encode_worker(self) -> None:
        try:
            while True:
                slot = self.completed_slots.get()
                if slot is None:
                    break
                self.writer.write(slot.host_overlay_array)
                self.completed_frames += 1
                if self.completed_frames % 30 == 0:
                    elapsed = time.perf_counter() - self.processing_started_at
                    fps = self.completed_frames / elapsed if elapsed > 0.0 else 0.0
                    print(
                        f"Processed {self.completed_frames}/"
                        f"{self.total_frames_display} frames | "
                        f"elapsed {elapsed:.2f} s | {fps:.2f} FPS"
                    )
                self.free_slots.put(slot)
        except BaseException as exc:
            self.errors.put(exc)
            self.stop_event.set()
        finally:
            try:
                self.writer.close()
            except BaseException as exc:
                self.errors.put(exc)
                self.stop_event.set()

    def _process_decoded_frames(self) -> None:
        while True:
            if not self.errors.empty():
                raise self.errors.get()
            try:
                slot = self.decoded_slots.get(timeout=0.1)
            except queue.Empty:
                if not self.errors.empty():
                    raise self.errors.get()
                continue
            if slot is None:
                break
            self.runner.process_video_slot(slot, self.alpha)
            self.completed_slots.put(slot)

    def run(self) -> None:
        """Start the workers, process every decoded frame, and report true FPS."""
        print(f"Decoder: {self.reader.description}")
        print(f"Encoder: {self.writer.description}")
        print(
            "Pipeline: one TensorRT context, one CUDA stream, "
            f"{self.free_slots.qsize()} bounded pinned host slots"
        )

        self.processing_started_at = time.perf_counter()
        decode_thread = threading.Thread(
            target=self._decode_worker, name="video-decode", daemon=True
        )
        encode_thread = threading.Thread(
            target=self._encode_worker, name="video-encode", daemon=True
        )
        decode_thread.start()
        encode_thread.start()
        try:
            self._process_decoded_frames()
        finally:
            self.stop_event.set()
            self.completed_slots.put(None)
            decode_thread.join(timeout=10.0)
            encode_thread.join(timeout=30.0)

        if decode_thread.is_alive() or encode_thread.is_alive():
            raise RuntimeError("A video pipeline worker did not stop cleanly.")
        if not self.errors.empty():
            raise self.errors.get()

        processing_elapsed = time.perf_counter() - self.processing_started_at
        average_processing_fps = (
            self.completed_frames / processing_elapsed
            if processing_elapsed > 0.0
            else 0.0
        )
        print(
            "Video processing complete: "
            f"{self.completed_frames} frames in {processing_elapsed:.2f} seconds "
            f"({average_processing_fps:.2f} FPS). Output: {self.output_path}"
        )


def run_video(
    runner: TensorRTRunner,
    video_path: Path,
    output_path: Path,
    input_size: tuple[int, int],
    alpha: float,
    video_backend: str,
    writer_backend: str,
    pipeline_slots: int,
    max_frames: int | None,
    cuda_kernels_path: Path,
) -> None:
    """Build and execute the bounded end-to-end video pipeline."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if pipeline_slots < 2:
        raise ValueError("--pipeline-slots must be at least 2.")

    info = probe_video(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = create_video_reader(video_backend, video_path, info, pipeline_slots)
    writer = create_video_writer(
        writer_backend,
        output_path,
        info,
        pipeline_slots,
        reader.channels,
    )

    input_h, input_w = input_size
    input_shape = (1, 3, input_h, input_w)
    frame_shape = (info.height, info.width, reader.channels)
    cuda_video_kernels = CudaVideoKernels(cuda_kernels_path)
    slots = runner.create_video_slots(
        input_shape,
        frame_shape,
        pipeline_slots,
        cuda_video_kernels,
    )

    VideoPipeline(
        runner=runner,
        reader=reader,
        writer=writer,
        slots=slots,
        info=info,
        output_path=output_path,
        alpha=alpha,
        max_frames=max_frames,
    ).run()
