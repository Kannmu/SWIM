from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .config import ProjectConfig
from .types import DICResult, FieldStatistics, PreprocessResult


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    if not np.isnan(arr).any():
        return arr
    mask = np.isnan(arr)
    arr[mask] = np.nanmedian(arr)
    return arr


def _bandpass_cube(cube: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    nyq = 0.5 * fs
    b, a = signal.butter(3, [low / nyq, high / nyq], btype="bandpass")
    return signal.filtfilt(b, a, cube, axis=0)


def _phase_map(cube: np.ndarray) -> np.ndarray:
    analytic = signal.hilbert(cube, axis=0)
    return np.angle(np.mean(analytic, axis=0))


def _compute_strain(mean_u_um: np.ndarray, mean_v_um: np.ndarray, step_px: float, pixel_size_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = step_px * pixel_size_um
    du_dy, du_dx = np.gradient(mean_u_um, spacing, spacing)
    dv_dy, dv_dx = np.gradient(mean_v_um, spacing, spacing)
    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)
    return exx, eyy, exy


def analyze_fields(dic: DICResult, pre: PreprocessResult, fps: float, config: ProjectConfig) -> FieldStatistics:
    pixel_size_um = config.analysis.pixel_size_um
    u = _fill_nan(dic.u)
    v = _fill_nan(dic.v)

    u_um = u * pixel_size_um
    v_um = v * pixel_size_um
    total = np.sqrt(u_um ** 2 + v_um ** 2)

    u_bp = _bandpass_cube(u_um, fps, config.dic.bandpass_temporal_hz)
    v_bp = _bandpass_cube(v_um, fps, config.dic.bandpass_temporal_hz)
    total_bp = np.sqrt(u_bp ** 2 + v_bp ** 2)

    if config.analysis.smoothing_sigma_frames > 0:
        sigma = (config.analysis.smoothing_sigma_frames, 0.0, 0.0)
        u_bp = ndimage.gaussian_filter(u_bp, sigma=sigma)
        v_bp = ndimage.gaussian_filter(v_bp, sigma=sigma)
        total_bp = ndimage.gaussian_filter(total_bp, sigma=sigma)

    mean_u = np.mean(u_bp, axis=0)
    mean_v = np.mean(v_bp, axis=0)
    rms_u = np.sqrt(np.mean(u_bp ** 2, axis=0))
    rms_v = np.sqrt(np.mean(v_bp ** 2, axis=0))
    amp_u = 0.5 * (np.max(u_bp, axis=0) - np.min(u_bp, axis=0))
    amp_v = 0.5 * (np.max(v_bp, axis=0) - np.min(v_bp, axis=0))
    amp_total = 0.5 * (np.max(total_bp, axis=0) - np.min(total_bp, axis=0))
    temporal_std = np.std(total_bp, axis=0)
    phase = _phase_map(total_bp)

    exx, eyy, exy = _compute_strain(mean_u, mean_v, dic.grid.step_size_px, pixel_size_um)

    center_y = len(dic.grid.centers_y) // 2
    center_x = len(dic.grid.centers_x) // 2
    center_trace = total_bp[:, center_y, center_x]
    ref_motion = np.linalg.norm(pre.reference_region_motion * pixel_size_um, axis=1)
    ref_corrected = center_trace - np.nan_to_num(ref_motion)
    time_s = np.arange(total_bp.shape[0], dtype=np.float64) / fps

    axis = config.analysis.spatial_wave_axis
    if axis == "auto":
        axis = "x" if np.nanmean(amp_u) >= np.nanmean(amp_v) else "y"

    if axis == "x":
        profile = amp_total[center_y, :]
        positions = dic.grid.centers_x * pixel_size_um / 1000.0
    else:
        profile = amp_total[:, center_x]
        positions = dic.grid.centers_y * pixel_size_um / 1000.0

    return FieldStatistics(
        mean_u=mean_u,
        mean_v=mean_v,
        rms_u=rms_u,
        rms_v=rms_v,
        amp_u=amp_u,
        amp_v=amp_v,
        amp_total=amp_total,
        temporal_std=temporal_std,
        dominant_phase=phase,
        strain_xx=exx,
        strain_yy=eyy,
        strain_xy=exy,
        wave_profile_axis=axis,
        wave_profile_positions_mm=positions,
        wave_profile_amplitude_um=profile,
        center_displacement_um=center_trace,
        center_time_s=time_s,
        ref_corrected_center_um=ref_corrected,
    )
