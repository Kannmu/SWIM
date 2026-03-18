from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import cv2

from .config import ProjectConfig
from .io_utils import ensure_parent, save_metadata
from .types import CaptureMetadata


class VideoCaptureSession:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def record(self, duration_s: float) -> CaptureMetadata:
        camera_cfg = self.config.camera
        strobe_cfg = self.config.strobe
        video_path = Path(self.config.paths.raw_video)
        ensure_parent(video_path)

        backend = camera_cfg.backend if camera_cfg.backend is not None else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_cfg.camera_index, backend)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开相机索引 {camera_cfg.camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
        cap.set(cv2.CAP_PROP_FPS, camera_cfg.fps)
        if camera_cfg.exposure_us is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(camera_cfg.exposure_us))
        if camera_cfg.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, float(camera_cfg.gain))

        fourcc = cv2.VideoWriter_fourcc(*camera_cfg.codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, camera_cfg.fps, (camera_cfg.width, camera_cfg.height), camera_cfg.color)
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"无法写入视频文件: {video_path}")

        started = dt.datetime.now().isoformat()
        start_time = time.perf_counter()
        frame_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not camera_cfg.color and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            writer.write(frame)
            frame_count += 1
            if time.perf_counter() - start_time >= duration_s:
                break

        writer.release()
        cap.release()
        ended = dt.datetime.now().isoformat()

        metadata = CaptureMetadata(
            width=camera_cfg.width,
            height=camera_cfg.height,
            fps=float(camera_cfg.fps),
            frame_count=frame_count,
            codec=camera_cfg.codec,
            is_color=camera_cfg.color,
            started_at=started,
            ended_at=ended,
            wave_frequency_hz=strobe_cfg.wave_frequency_hz,
            strobe_frequency_hz=strobe_cfg.strobe_frequency_hz,
            target_beat_hz=strobe_cfg.target_beat_hz,
            pulse_width_us=strobe_cfg.pulse_width_us,
            notes=self.config.notes,
            video_path=video_path,
        )
        save_metadata(metadata, self.config.paths.metadata_json)
        return metadata
