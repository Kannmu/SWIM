from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .analysis import analyze_fields
from .capture import VideoCaptureSession
from .config import ProjectConfig, load_config
from .dic_core import run_dic
from .io_utils import load_video_gray, save_frames_png, save_gray_video, save_npz
from .preprocess import preprocess_frames
from .types import CaptureMetadata, PipelineArtifacts
from .visualization import export_visualizations

logger = logging.getLogger("swim_dic")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _select_bright_frames(sequence, config: ProjectConfig):
    frame_means = sequence.frames.mean(axis=(1, 2)).astype(np.float32)
    low = float(np.min(frame_means))
    high = float(np.max(frame_means))
    threshold = 0.5 * (low + high)
    bright_mask = frame_means >= threshold
    if not np.any(bright_mask):
        raise RuntimeError("亮帧提取失败：未检测到亮帧")
    bright_indices = np.flatnonzero(bright_mask)
    bright_frames = sequence.frames[bright_indices]
    bright_fps = (
        float(config.analysis.pure_bright_video_fps)
        if config.analysis.pure_bright_video_fps is not None
        else float(config.strobe.strobe_frequency_hz)
    )
    timestamps = np.arange(bright_frames.shape[0], dtype=np.float64) / bright_fps
    logger.info(
        "Pipeline: 亮帧提取完成, bright_frames=%d/%d, threshold=%.3f, bright_fps=%.3f",
        bright_frames.shape[0],
        sequence.frames.shape[0],
        threshold,
        bright_fps,
    )
    return bright_indices, frame_means, threshold, bright_frames, bright_fps, timestamps


def run_analysis_pipeline(config: ProjectConfig) -> PipelineArtifacts:
    logger.info("Pipeline: 开始分析, raw_video=%s", config.paths.raw_video)
    sequence = load_video_gray(Path(config.paths.raw_video))
    logger.info(
        "Pipeline: 视频加载完成, frames=%d, fps=%.3f, shape=%s",
        sequence.frames.shape[0],
        sequence.fps,
        sequence.frames.shape[1:],
    )
    if config.paths.frames_dir:
        logger.info("Pipeline: 导出原始帧 PNG 到 %s", config.paths.frames_dir)
        save_frames_png(sequence, config.paths.frames_dir)

    bright_indices, frame_means, bright_threshold, bright_frames, bright_fps, bright_timestamps = _select_bright_frames(sequence, config)
    bright_video_path = Path(config.paths.bright_video)
    if config.analysis.export_pure_bright_video:
        logger.info("Pipeline: 写出纯亮帧视频到 %s", bright_video_path)
        save_gray_video(bright_frames, bright_video_path, bright_fps, codec=config.camera.codec)

    sequence = type(sequence)(frames=bright_frames, fps=bright_fps, timestamps_s=bright_timestamps)

    pre = preprocess_frames(sequence.frames, config)
    dic = run_dic(pre, config)
    stats = analyze_fields(dic, pre, sequence.fps, config)

    logger.info("Pipeline: 导出图像、CSV 与视频结果到 %s", config.paths.output_dir)
    export_visualizations(pre, dic, stats, config)

    if config.analysis.export_npz:
        logger.info("Pipeline: 写出 NPZ 结果文件")
        save_npz(
            Path(config.paths.output_dir) / "dic_results.npz",
            u=dic.u,
            v=dic.v,
            corr=dic.corr,
            centers_x=dic.grid.centers_x,
            centers_y=dic.grid.centers_y,
            amp_total=stats.amp_total,
            center_time_s=stats.center_time_s,
            center_displacement_um=stats.center_displacement_um,
            ref_corrected_center_um=stats.ref_corrected_center_um,
            center_strain_xy=stats.center_strain_xy,
            reference_region_motion=pre.reference_region_motion,
            rigid_transforms=pre.rigid_transforms,
            xt_displacement=stats.xt_displacement,
            xt_strain_xy=stats.xt_strain_xy,
            wavefront_phase_indices=stats.wavefront_phase_indices,
            wavefront_displacement_snapshots=stats.wavefront_displacement_snapshots,
            wavefront_strain_xy_snapshots=stats.wavefront_strain_xy_snapshots,
            signed_wave_field=stats.signed_wave_field,
            roi_offset_xy=pre.roi_offset_xy,
            diagnostics_total_matches=dic.diagnostics.total_matches,
            diagnostics_num_points=dic.diagnostics.num_points,
            bright_frame_indices=bright_indices,
            bright_frame_means=frame_means,
            bright_threshold=bright_threshold,
        )

    metadata_path = Path(config.paths.metadata_json)
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = CaptureMetadata(
            width=payload["width"],
            height=payload["height"],
            fps=payload["fps"],
            frame_count=payload["frame_count"],
            codec=payload["codec"],
            is_color=payload["is_color"],
            started_at=payload["started_at"],
            ended_at=payload["ended_at"],
            wave_frequency_hz=payload["wave_frequency_hz"],
            strobe_frequency_hz=payload["strobe_frequency_hz"],
            target_beat_hz=payload["target_beat_hz"],
            pulse_width_us=payload["pulse_width_us"],
            notes=payload.get("notes", ""),
            video_path=Path(payload["video_path"]),
            bright_video_path=Path(payload["bright_video_path"]) if payload.get("bright_video_path") else bright_video_path,
            bright_frame_count=payload.get("bright_frame_count", int(bright_frames.shape[0])),
            bright_video_fps=payload.get("bright_video_fps", float(bright_fps)),
        )
        payload["bright_video_path"] = str(bright_video_path)
        payload["bright_frame_count"] = int(bright_frames.shape[0])
        payload["bright_video_fps"] = float(bright_fps)
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        metadata = CaptureMetadata(
            width=sequence.frames.shape[2],
            height=sequence.frames.shape[1],
            fps=sequence.fps,
            frame_count=sequence.frames.shape[0],
            codec="unknown",
            is_color=False,
            started_at="",
            ended_at="",
            wave_frequency_hz=config.strobe.wave_frequency_hz,
            strobe_frequency_hz=config.strobe.strobe_frequency_hz,
            target_beat_hz=config.strobe.target_beat_hz,
            pulse_width_us=config.strobe.pulse_width_us,
            notes=config.notes,
            video_path=Path(config.paths.raw_video),
            bright_video_path=bright_video_path,
            bright_frame_count=int(bright_frames.shape[0]),
            bright_video_fps=float(bright_fps),
        )

    logger.info("Pipeline: 分析完成")
    return PipelineArtifacts(metadata=metadata, preprocessing=pre, dic=dic, stats=stats)


def capture_and_run(config_path: str | Path, duration_s: float) -> PipelineArtifacts:
    config = load_config(config_path)
    session = VideoCaptureSession(config)
    logger.info("Pipeline: 先录制后分析, duration_s=%.3f", duration_s)
    session.record(duration_s)
    return run_analysis_pipeline(config)
