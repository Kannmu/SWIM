from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import CaptureMetadata, FrameSequence


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_metadata(metadata: CaptureMetadata, path: Path) -> None:
    ensure_parent(path)
    payload = asdict(metadata)
    payload["video_path"] = str(metadata.video_path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_video_gray(video_path: Path) -> FrameSequence:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        frames.append(gray.astype(np.float32))
    capture.release()

    if not frames:
        raise RuntimeError(f"视频为空: {video_path}")

    array = np.stack(frames, axis=0)
    timestamps = np.arange(array.shape[0], dtype=np.float64) / float(fps)
    return FrameSequence(frames=array, fps=float(fps), timestamps_s=timestamps)


def save_gray_video(frames: np.ndarray, path: Path, fps: float, codec: str = "mp4v") -> None:
    if frames.ndim != 3:
        raise ValueError("保存视频要求 frames 形状为 (n, h, w)")
    if frames.shape[0] == 0:
        raise ValueError("保存视频失败：没有可写入的帧")
    ensure_parent(path)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height), False)
    if not writer.isOpened():
        raise RuntimeError(f"无法写入视频文件: {path}")
    try:
        for frame in frames:
            writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    finally:
        writer.release()


def save_frames_png(sequence: FrameSequence, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(sequence.frames):
        path = frames_dir / f"frame_{idx:05d}.png"
        cv2.imwrite(str(path), np.clip(frame, 0, 255).astype(np.uint8))


def save_npz(path: Path, **payload: Any) -> None:
    ensure_parent(path)
    np.savez_compressed(path, **payload)
