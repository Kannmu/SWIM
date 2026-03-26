from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator, field_validator, field_validator


class ReferenceRegion(BaseModel):
    name: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    weight: float = Field(default=1.0, ge=0.0)


class CameraConfig(BaseModel):
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 60
    disable_auto_exposure: bool = True
    exposure_us: float | None = None
    gain: float | None = None
    codec: str = "mp4v"
    color: bool = True
    backend: int | None = None


class StrobeConfig(BaseModel):
    wave_frequency_hz: float = 200.0
    strobe_frequency_hz: float = 49.75
    target_beat_hz: float = 1.0
    pulse_width_us: float = 10.0
    led_color: str = "green"


class DICConfig(BaseModel):
    roi: tuple[int, int, int, int] | None = None
    subset_size_px: int = Field(default=31, ge=9)
    step_size_px: int = Field(default=8, ge=1)
    search_radius_px: int = Field(default=12, ge=2)
    gaussian_sigma_px: float = Field(default=0.8, ge=0.0)
    highpass_sigma_px: float = Field(default=9.0, ge=0.0)
    clahe_clip_limit: float = Field(default=2.0, ge=0.1)
    clahe_tile_grid_size: int = Field(default=8, ge=2)
    bandpass_temporal_hz: tuple[float, float] = (0.2, 5.0)
    global_motion_model: Literal["euclidean", "affine"] = "euclidean"
    outlier_mad_scale: float = Field(default=4.5, ge=1.0)
    reference_strategy: Literal["first_frame", "median"] = "median"
    use_reference_region_correction: bool = True
    use_global_motion_correction: bool = True
    use_common_mode_subtraction: bool = True
    median_reference_frame_count: int = Field(default=21, ge=3)
    enable_gpu: bool = True
    gpu_backend: Literal["auto", "cupy"] = "auto"
    gpu_batch_size: int = Field(default=8, ge=1)
    numba_parallel: bool = True

    @model_validator(mode="after")
    def validate_sizes(self) -> "DICConfig":
        if self.subset_size_px % 2 == 0:
            raise ValueError("subset_size_px 必须为奇数")
        if self.search_radius_px < self.step_size_px:
            raise ValueError("search_radius_px 不能小于 step_size_px")
        low, high = self.bandpass_temporal_hz
        if low < 0 or high <= low:
            raise ValueError("bandpass_temporal_hz 必须满足 0 <= low < high")
        return self


class AnalysisConfig(BaseModel):
    expected_wave_frequency_hz: float = 200.0
    equivalent_slow_frequency_hz: float = 1.0
    pure_bright_video_fps: float | None = None
    export_pure_bright_video: bool = True
    pixel_size_um: float = Field(default=10.0, gt=0)
    beat_cycles_to_analyze: int = Field(default=1, ge=1)
    smoothing_sigma_frames: float = Field(default=1.0, ge=0.0)
    export_video_overlays: bool = True
    export_field_csv: bool = True
    export_npz: bool = True
    visualize_quiver_stride: int = Field(default=3, ge=1)
    spatial_wave_axis: Literal["x", "y", "auto"] = "auto"


class CalibrationBoardConfig(BaseModel):
    inner_corners_rows: int = Field(default=9, ge=2)
    inner_corners_cols: int = Field(default=12, ge=2)
    square_size_mm: float = Field(default=15.0, gt=0)


class CalibrationConfig(BaseModel):
    enabled: bool = False
    board: CalibrationBoardConfig = Field(default_factory=CalibrationBoardConfig)
    camera_matrix: list[list[float]] = Field(default_factory=list)
    distortion_coefficients: list[float] = Field(default_factory=list)
    optimal_camera_matrix: list[list[float]] = Field(default_factory=list)
    roi: list[int] = Field(default_factory=list)
    image_size: list[int] = Field(default_factory=list)
    mean_reprojection_error_px: float | None = None
    rms_reprojection_error_px: float | None = None
    pixel_size_um: float | None = None
    pixel_size_std_um: float | None = None


class PathsConfig(BaseModel):
    project_root: Path = Path(".")
    raw_video: Path = Path("data/raw/capture.mp4")
    bright_video: Path = Path("data/derived/bright_only.mp4")
    frames_dir: Path = Path("data/frames")
    output_dir: Path = Path("outputs")
    metadata_json: Path = Path("data/raw/capture_metadata.json")

    @field_validator("project_root", "raw_video", "bright_video", "frames_dir", "output_dir", "metadata_json", mode="before")
    @classmethod
    def normalize_paths(cls, v):
        if isinstance(v, str):
            return v.replace("\\", "/")
        return v

    @field_validator("project_root", "raw_video", "bright_video", "frames_dir", "output_dir", "metadata_json", mode="before")
    @classmethod
    def normalize_paths(cls, v):
        if isinstance(v, str):
            return v.replace("\\", "/")
        return v


class ProjectConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    strobe: StrobeConfig = Field(default_factory=StrobeConfig)
    dic: DICConfig = Field(default_factory=DICConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    reference_regions: list[ReferenceRegion] = Field(default_factory=list)
    notes: str = ""

    def resolve_paths(self, config_path: Path | None = None) -> "ProjectConfig":
        base = config_path.parent if config_path else Path.cwd()
        if not self.paths.project_root.is_absolute():
            self.paths.project_root = (base / self.paths.project_root).resolve()
        for attr in ["raw_video", "bright_video", "frames_dir", "output_dir", "metadata_json"]:
            value = getattr(self.paths, attr)
            if not value.is_absolute():
                setattr(self.paths, attr, (self.paths.project_root / value).resolve())
        return self


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = ProjectConfig.model_validate(payload)
    return config.resolve_paths(config_path)


def save_config_template(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig()
    text = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    target.write_text(text, encoding="utf-8")
    return target
