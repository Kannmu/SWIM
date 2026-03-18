from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CaptureMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    is_color: bool
    started_at: str
    ended_at: str
    wave_frequency_hz: float
    strobe_frequency_hz: float
    target_beat_hz: float
    pulse_width_us: float
    notes: str
    video_path: Path


@dataclass(slots=True)
class FrameSequence:
    frames: np.ndarray
    fps: float
    timestamps_s: np.ndarray


@dataclass(slots=True)
class PreprocessResult:
    raw_frames: np.ndarray
    processed_frames: np.ndarray
    reference_frame: np.ndarray
    rigid_transforms: np.ndarray
    reference_region_motion: np.ndarray
    common_mode_signal: np.ndarray


@dataclass(slots=True)
class GridDefinition:
    centers_x: np.ndarray
    centers_y: np.ndarray
    subset_size_px: int
    step_size_px: int


@dataclass(slots=True)
class DICResult:
    u: np.ndarray
    v: np.ndarray
    corr: np.ndarray
    grid: GridDefinition


@dataclass(slots=True)
class FieldStatistics:
    mean_u: np.ndarray
    mean_v: np.ndarray
    rms_u: np.ndarray
    rms_v: np.ndarray
    amp_u: np.ndarray
    amp_v: np.ndarray
    amp_total: np.ndarray
    temporal_std: np.ndarray
    dominant_phase: np.ndarray
    strain_xx: np.ndarray
    strain_yy: np.ndarray
    strain_xy: np.ndarray
    wave_profile_axis: str
    wave_profile_positions_mm: np.ndarray
    wave_profile_amplitude_um: np.ndarray
    center_displacement_um: np.ndarray
    center_time_s: np.ndarray
    ref_corrected_center_um: np.ndarray


@dataclass(slots=True)
class PipelineArtifacts:
    metadata: CaptureMetadata
    preprocessing: PreprocessResult
    dic: DICResult
    stats: FieldStatistics
