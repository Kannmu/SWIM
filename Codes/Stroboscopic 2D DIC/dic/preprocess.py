from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage, signal

from .config import ProjectConfig, ReferenceRegion
from .types import PreprocessResult


def _clahe_u8(frame: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(np.clip(frame, 0, 255).astype(np.uint8)).astype(np.float32)


def _bandpass_frame(frame: np.ndarray, sigma_low: float, sigma_high: float) -> np.ndarray:
    low = ndimage.gaussian_filter(frame, sigma=sigma_low) if sigma_low > 0 else frame
    high = frame - ndimage.gaussian_filter(frame, sigma=sigma_high) if sigma_high > 0 else frame
    return high + 0.2 * low


def _estimate_euclidean(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        cv2.findTransformECC(reference, moving, warp, cv2.MOTION_EUCLIDEAN, criteria)
    except cv2.error:
        pass
    return warp


def _apply_warp(frame: np.ndarray, warp: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.warpAffine(
        frame,
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


def _region_mean_motion(frames: np.ndarray, regions: list[ReferenceRegion]) -> np.ndarray:
    if not regions:
        return np.zeros((frames.shape[0], 2), dtype=np.float32)

    tracks = []
    ref = frames[0]
    for region in regions:
        x, y, w, h = region.x, region.y, region.width, region.height
        tpl = ref[y:y + h, x:x + w]
        center = np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)
        track = []
        for frame in frames:
            res = cv2.matchTemplate(frame.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            current = np.array([max_loc[0] + 0.5 * w, max_loc[1] + 0.5 * h], dtype=np.float32)
            track.append((current - center) * region.weight)
        tracks.append(np.asarray(track, dtype=np.float32))
    return np.sum(tracks, axis=0) / max(sum(r.weight for r in regions), 1e-6)


def preprocess_frames(frames: np.ndarray, config: ProjectConfig) -> PreprocessResult:
    dic_cfg = config.dic
    work = frames.astype(np.float32)

    if dic_cfg.roi is not None:
        x, y, w, h = dic_cfg.roi
        work = work[:, y:y + h, x:x + w]

    enhanced = np.stack(
        [
            _bandpass_frame(
                _clahe_u8(frame, dic_cfg.clahe_clip_limit, dic_cfg.clahe_tile_grid_size),
                dic_cfg.gaussian_sigma_px,
                dic_cfg.highpass_sigma_px,
            )
            for frame in work
        ],
        axis=0,
    )

    if dic_cfg.reference_strategy == "median":
        n_ref = min(dic_cfg.median_reference_frame_count, enhanced.shape[0])
        reference_frame = np.median(enhanced[:n_ref], axis=0).astype(np.float32)
    else:
        reference_frame = enhanced[0].copy()

    warps = []
    aligned = []
    for frame in enhanced:
        warp = _estimate_euclidean(reference_frame, frame) if dic_cfg.use_global_motion_correction else np.eye(2, 3, dtype=np.float32)
        warps.append(warp)
        aligned.append(_apply_warp(frame, warp))
    aligned = np.stack(aligned, axis=0)
    warps = np.stack(warps, axis=0)

    ref_motion = _region_mean_motion(aligned, config.reference_regions) if dic_cfg.use_reference_region_correction else np.zeros((aligned.shape[0], 2), dtype=np.float32)

    common_mode = np.zeros(aligned.shape[0], dtype=np.float32)
    if dic_cfg.use_common_mode_subtraction:
        spatial_mean = aligned.mean(axis=(1, 2))
        common_mode = signal.detrend(spatial_mean, type="linear").astype(np.float32)
        aligned = aligned - common_mode[:, None, None]

    return PreprocessResult(
        raw_frames=work,
        processed_frames=aligned,
        reference_frame=reference_frame,
        rigid_transforms=warps,
        reference_region_motion=ref_motion,
        common_mode_signal=common_mode,
    )
