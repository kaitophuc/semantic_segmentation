"""Video decoder and encoder backends used by the inference pipeline."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np

from inference_helpers import VideoInfo


class VideoReader(Protocol):
    channels: int
    description: str

    def read_into(self, destination: np.ndarray) -> bool: ...

    def close(self) -> None: ...


class VideoWriter(Protocol):
    description: str

    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


class OpenCVVideoReader:
    channels = 3
    description = "OpenCV/FFmpeg software decode"

    def __init__(self, video_path: Path) -> None:
        self.capture = cv.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(
                f"Video not found or could not be opened: {video_path}"
            )

    def read_into(self, destination: np.ndarray) -> bool:
        ok, frame = self.capture.read()
        if not ok:
            return False
        np.copyto(destination, frame)
        return True

    def close(self) -> None:
        self.capture.release()


class OpenCVVideoWriter:
    description = "OpenCV MPEG-4 software encode"

    def __init__(self, output_path: Path, info: VideoInfo) -> None:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        self.writer = cv.VideoWriter(
            str(output_path), fourcc, info.fps, (info.width, info.height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame[:, :, :3])

    def close(self) -> None:
        self.writer.release()


_GSTREAMER_MODULES: tuple[object, object, object] | None = None


def _load_gstreamer() -> tuple[object, object, object]:
    global _GSTREAMER_MODULES
    if _GSTREAMER_MODULES is not None:
        return _GSTREAMER_MODULES

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import GLib, Gst, GstVideo

    Gst.init(None)
    _GSTREAMER_MODULES = Gst, GstVideo, GLib
    return _GSTREAMER_MODULES


def _require_gstreamer_elements(*names: str) -> None:
    Gst, _, _ = _load_gstreamer()
    missing = [name for name in names if Gst.ElementFactory.find(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required GStreamer elements: " + ", ".join(missing)
        )


def _gstreamer_error(bus: object, timeout: int = 0) -> RuntimeError | None:
    Gst, _, _ = _load_gstreamer()
    message = bus.timed_pop_filtered(timeout, Gst.MessageType.ERROR)
    if message is None:
        return None
    error, debug = message.parse_error()
    return RuntimeError(f"GStreamer error: {error.message} ({debug})")


class GStreamerVideoReader:
    channels = 4
    description = "Jetson NVDEC HEVC decode + VIC BGRx conversion"

    def __init__(self, video_path: Path, info: VideoInfo, queue_size: int) -> None:
        Gst, _, _ = _load_gstreamer()
        _require_gstreamer_elements(
            "qtdemux", "h265parse", "nvv4l2decoder", "nvvidconv", "appsink"
        )
        location = str(video_path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
        pipeline_text = (
            f'filesrc location="{location}" ! qtdemux ! h265parse ! '
            "nvv4l2decoder enable-max-performance=true ! "
            "nvvidconv compute-hw=2 ! video/x-raw,format=BGRx ! "
            f"appsink name=framesink sync=false drop=false max-buffers={queue_size}"
        )
        self.pipeline = Gst.parse_launch(pipeline_text)
        self.sink = self.pipeline.get_by_name("framesink")
        self.bus = self.pipeline.get_bus()
        self.info = info
        state = self.pipeline.set_state(Gst.State.PLAYING)
        if state == Gst.StateChangeReturn.FAILURE:
            error = _gstreamer_error(self.bus, Gst.SECOND)
            self.pipeline.set_state(Gst.State.NULL)
            if error is not None:
                raise error
            raise RuntimeError("Could not start the Jetson GStreamer decoder.")

    def read_into(self, destination: np.ndarray) -> bool:
        Gst, GstVideo, _ = _load_gstreamer()
        sample = self.sink.emit("pull-sample")
        if sample is None:
            error = _gstreamer_error(self.bus)
            if error is not None:
                raise error
            return False

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        video_info = GstVideo.VideoInfo()
        if not video_info.from_caps(caps):
            raise RuntimeError(f"Could not read GStreamer video caps: {caps.to_string()}")
        mapped, map_info = buffer.map(Gst.MapFlags.READ)
        if not mapped:
            raise RuntimeError("Could not map a decoded GStreamer frame.")
        try:
            stride = int(video_info.stride[0])
            if stride == 0:
                stride = buffer.get_size() // self.info.height
            if stride < self.info.width * self.channels:
                raise RuntimeError(
                    f"Decoded frame stride {stride} is smaller than its visible width."
                )
            mapped_frame = np.ndarray(
                (self.info.height, stride // self.channels, self.channels),
                dtype=np.uint8,
                buffer=map_info.data,
            )
            np.copyto(destination, mapped_frame[:, : self.info.width, :])
        finally:
            buffer.unmap(map_info)
        return True

    def close(self) -> None:
        Gst, _, _ = _load_gstreamer()
        self.pipeline.set_state(Gst.State.NULL)


class GStreamerVideoWriter:
    description = "VIC color conversion + x264 ultrafast software H.264 encode"

    def __init__(self, output_path: Path, info: VideoInfo, queue_size: int) -> None:
        Gst, _, _ = _load_gstreamer()
        _require_gstreamer_elements(
            "appsrc", "nvvidconv", "x264enc", "h264parse", "mp4mux", "filesink"
        )
        location = str(output_path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
        fps = Fraction(info.fps).limit_denominator(1001)
        self.fps_numerator = fps.numerator
        self.fps_denominator = fps.denominator
        pipeline_text = (
            "appsrc name=framesource is-live=false block=true format=time "
            f"caps=video/x-raw,format=BGRx,width={info.width},height={info.height},"
            f"framerate={fps.numerator}/{fps.denominator} ! "
            f"queue max-size-buffers={queue_size} ! "
            "nvvidconv compute-hw=2 ! video/x-raw,format=I420 ! "
            "x264enc speed-preset=ultrafast tune=zerolatency bitrate=12000 "
            "key-int-max=30 bframes=0 threads=6 ! h264parse ! mp4mux ! "
            f'filesink location="{location}"'
        )
        self.pipeline = Gst.parse_launch(pipeline_text)
        self.source = self.pipeline.get_by_name("framesource")
        self.bus = self.pipeline.get_bus()
        self.frame_index = 0
        state = self.pipeline.set_state(Gst.State.PLAYING)
        if state == Gst.StateChangeReturn.FAILURE:
            error = _gstreamer_error(self.bus, Gst.SECOND)
            self.pipeline.set_state(Gst.State.NULL)
            if error is not None:
                raise error
            raise RuntimeError("Could not start the GStreamer output pipeline.")

    def write(self, frame: np.ndarray) -> None:
        Gst, _, _ = _load_gstreamer()
        frame_bytes = frame.tobytes(order="C")
        buffer = Gst.Buffer.new_wrapped(frame_bytes)
        buffer.pts = Gst.util_uint64_scale(
            self.frame_index,
            Gst.SECOND * self.fps_denominator,
            self.fps_numerator,
        )
        buffer.dts = buffer.pts
        buffer.duration = Gst.util_uint64_scale(
            1, Gst.SECOND * self.fps_denominator, self.fps_numerator
        )
        result = self.source.emit("push-buffer", buffer)
        if result != Gst.FlowReturn.OK:
            error = _gstreamer_error(self.bus)
            if error is not None:
                raise error
            raise RuntimeError(f"GStreamer writer returned {result}")
        self.frame_index += 1

    def close(self) -> None:
        Gst, _, _ = _load_gstreamer()
        result = self.source.emit("end-of-stream")
        if result != Gst.FlowReturn.OK:
            raise RuntimeError(f"Could not finish GStreamer output: {result}")
        message = self.bus.timed_pop_filtered(
            30 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS
        )
        try:
            if message is None:
                raise RuntimeError("Timed out while finalizing the GStreamer output.")
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"GStreamer error: {error.message} ({debug})")
        finally:
            self.pipeline.set_state(Gst.State.NULL)


def create_video_reader(
    backend: str,
    video_path: Path,
    info: VideoInfo,
    queue_size: int,
) -> VideoReader:
    if backend in ("auto", "gstreamer"):
        try:
            return GStreamerVideoReader(video_path, info, queue_size)
        except Exception as exc:
            if backend == "gstreamer":
                raise
            print(f"GStreamer decoder unavailable ({exc}); using OpenCV.")
    return OpenCVVideoReader(video_path)


def create_video_writer(
    backend: str,
    output_path: Path,
    info: VideoInfo,
    queue_size: int,
    channels: int,
) -> VideoWriter:
    if backend == "gstreamer" and channels != 4:
        raise RuntimeError("The GStreamer writer requires BGRx (four-channel) frames.")
    if backend in ("auto", "gstreamer") and channels == 4:
        try:
            return GStreamerVideoWriter(output_path, info, queue_size)
        except Exception as exc:
            if backend == "gstreamer":
                raise
            print(f"GStreamer writer unavailable ({exc}); using OpenCV.")
    return OpenCVVideoWriter(output_path, info)
