from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage, signal

from .config import ProjectConfig
from .types import DICResult, FieldStatistics, PreprocessResult

logger = logging.getLogger("swim_dic")


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


def _compute_dynamic_strain(u_cube_um: np.ndarray, v_cube_um: np.ndarray, step_px: float, pixel_size_um: float) -> np.ndarray:
    spacing = step_px * pixel_size_um
    du_dy = np.gradient(u_cube_um, spacing, axis=1)
    dv_dx = np.gradient(v_cube_um, spacing, axis=2)
    return 0.5 * (du_dy + dv_dx)


def _component_statistics(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(cube, axis=0)
    rms = np.sqrt(np.mean(np.square(cube), axis=0))
    amp = 0.5 * np.ptp(cube, axis=0)
    return mean, rms, amp


def _phase_indices(nt: int) -> np.ndarray:
    if nt <= 1:
        return np.array([0], dtype=np.int32)
    raw = np.linspace(0, nt - 1, 4)
    return np.unique(np.round(raw).astype(np.int32))


def analyze_fields(dic: DICResult, pre: PreprocessResult, fps: float, config: ProjectConfig) -> FieldStatistics:
    pixel_size_um = config.analysis.pixel_size_um
    logger.info("分析: 开始统计位移场, fps=%.3f, pixel_size_um=%.6f", fps, pixel_size_um)
    u = _fill_nan(dic.u)
    v = _fill_nan(dic.v)

    uv = np.stack((u, v), axis=0).astype(np.float32, copy=False)
    uv_um = uv * np.float32(pixel_size_um)

    logger.info("分析: 开始时间带通滤波, band=%s", config.dic.bandpass_temporal_hz)
    u_bp = _bandpass_cube(uv_um[0], fps, config.dic.bandpass_temporal_hz).astype(np.float32, copy=False)
    v_bp = _bandpass_cube(uv_um[1], fps, config.dic.bandpass_temporal_hz).astype(np.float32, copy=False)
    uv_bp = np.stack((u_bp, v_bp), axis=0)
    total_bp = np.sqrt(np.sum(np.square(uv_bp), axis=0, dtype=np.float32), dtype=np.float32)

    if config.analysis.smoothing_sigma_frames > 0:
        sigma = (config.analysis.smoothing_sigma_frames, 0.0, 0.0)
        logger.info("分析: 对时间轴执行高斯平滑, sigma_frames=%.3f", config.analysis.smoothing_sigma_frames)
        u_bp = ndimage.gaussian_filter(u_bp, sigma=sigma)
        v_bp = ndimage.gaussian_filter(v_bp, sigma=sigma)
        total_bp = ndimage.gaussian_filter(total_bp, sigma=sigma)

    mean_u, rms_u, amp_u = _component_statistics(u_bp)
    mean_v, rms_v, amp_v = _component_statistics(v_bp)
    amp_total = 0.5 * np.ptp(total_bp, axis=0)
    temporal_std = np.std(total_bp, axis=0)
    phase = _phase_map(total_bp)

    exx, eyy, exy = _compute_strain(mean_u, mean_v, dic.grid.step_size_px, pixel_size_um)
    strain_xy_dynamic = _compute_dynamic_strain(u_bp, v_bp, dic.grid.step_size_px, pixel_size_um).astype(np.float32, copy=False)
    signed_wave_field = strain_xy_dynamic

    center_y = len(dic.grid.centers_y) // 2
    center_x = len(dic.grid.centers_x) // 2
    center_trace = total_bp[:, center_y, center_x]
    center_strain_xy = strain_xy_dynamic[:, center_y, center_x]
    ref_motion = np.linalg.norm(pre.reference_region_motion * pixel_size_um, axis=1)
    ref_corrected = center_trace - np.nan_to_num(ref_motion)
    time_s = np.arange(total_bp.shape[0], dtype=np.float64) / fps

    axis = config.analysis.spatial_wave_axis
    if axis == "auto":
        axis = "x" if np.nanmean(amp_u) >= np.nanmean(amp_v) else "y"

    if axis == "x":
        profile = amp_total[center_y, :]
        positions = dic.grid.centers_x * pixel_size_um / 1000.0
        xt_displacement = total_bp[:, center_y, :]
        xt_strain_xy = strain_xy_dynamic[:, center_y, :]
    else:
        profile = amp_total[:, center_x]
        positions = dic.grid.centers_y * pixel_size_um / 1000.0
        xt_displacement = total_bp[:, :, center_x]
        xt_strain_xy = strain_xy_dynamic[:, :, center_x]

    phase_indices = _phase_indices(total_bp.shape[0])
    wavefront_displacement_snapshots = total_bp[phase_indices]
    wavefront_strain_xy_snapshots = strain_xy_dynamic[phase_indices]
    logger.info(
        "分析: 完成, xt_shape=%s, snapshot_count=%d, axis=%s",
        xt_displacement.shape,
        len(phase_indices),
        axis,
    )

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
        signed_wave_field=signed_wave_field,
        center_strain_xy=center_strain_xy,
        xt_displacement=xt_displacement,
        xt_strain_xy=xt_strain_xy,
        wavefront_phase_indices=phase_indices,
        wavefront_displacement_snapshots=wavefront_displacement_snapshots,
        wavefront_strain_xy_snapshots=wavefront_strain_xy_snapshots,
        wave_profile_axis=axis,
        wave_profile_positions_mm=positions,
        wave_profile_amplitude_um=profile,
        center_displacement_um=center_trace,
        center_time_s=time_s,
        ref_corrected_center_um=ref_corrected,
    )
