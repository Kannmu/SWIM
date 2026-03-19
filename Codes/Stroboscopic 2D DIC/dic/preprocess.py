from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy import ndimage, signal

from .config import ProjectConfig, ReferenceRegion
from .types import PreprocessResult, UndistortionInfo

logger = logging.getLogger("swim_dic")


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


def _clip_region_to_shape(region: ReferenceRegion, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0 = int(np.clip(region.x, 0, max(0, w - 1)))
    y0 = int(np.clip(region.y, 0, max(0, h - 1)))
    x1 = int(np.clip(region.x + region.width, x0 + 1, w))
    y1 = int(np.clip(region.y + region.height, y0 + 1, h))
    return x0, y0, x1 - x0, y1 - y0


def _region_mean_motion(frames: np.ndarray, regions: list[ReferenceRegion], roi_offset_xy: tuple[int, int]) -> np.ndarray:
    if not regions:
        return np.zeros((frames.shape[0], 2), dtype=np.float32)

    tracks = []
    ref = frames[0]
    offset_x, offset_y = roi_offset_xy
    for region in regions:
        local_region = ReferenceRegion(
            name=region.name,
            x=region.x - offset_x,
            y=region.y - offset_y,
            width=region.width,
            height=region.height,
            weight=region.weight,
        )
        x, y, w, h = _clip_region_to_shape(local_region, ref.shape)
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


def _build_undistortion_info(config: ProjectConfig) -> UndistortionInfo:
    calib = config.calibration
    if not calib.enabled or not calib.camera_matrix or not calib.distortion_coefficients:
        return UndistortionInfo(
            applied=False,
            roi_xywh=None,
            camera_matrix=None,
            distortion_coefficients=None,
            optimal_camera_matrix=None,
        )
    roi_xywh = tuple(int(v) for v in calib.roi) if calib.roi else None
    optimal = np.asarray(calib.optimal_camera_matrix, dtype=np.float32) if calib.optimal_camera_matrix else None
    return UndistortionInfo(
        applied=True,
        roi_xywh=roi_xywh,
        camera_matrix=np.asarray(calib.camera_matrix, dtype=np.float32),
        distortion_coefficients=np.asarray(calib.distortion_coefficients, dtype=np.float32),
        optimal_camera_matrix=optimal,
    )


def _undistort_frames(frames: np.ndarray, info: UndistortionInfo) -> np.ndarray:
    if not info.applied or info.camera_matrix is None or info.distortion_coefficients is None:
        return frames
    camera_matrix = info.camera_matrix
    optimal_camera_matrix = info.optimal_camera_matrix if info.optimal_camera_matrix is not None else camera_matrix
    logger.info("预处理: 使用相机标定结果对视频逐帧去畸变")
    corrected = [
        cv2.undistort(frame.astype(np.float32), camera_matrix, info.distortion_coefficients, None, optimal_camera_matrix)
        for frame in frames
    ]
    return np.stack(corrected, axis=0).astype(np.float32, copy=False)


def preprocess_frames(frames: np.ndarray, config: ProjectConfig) -> PreprocessResult:
    dic_cfg = config.dic
    work = frames.astype(np.float32)
    undistortion = _build_undistortion_info(config)
    work = _undistort_frames(work, undistortion)

    roi_offset_xy = (0, 0)
    if dic_cfg.roi is not None:
        x, y, w, h = dic_cfg.roi
        roi_offset_xy = (x, y)
        work = work[:, y:y + h, x:x + w]
        logger.info("预处理: 应用 ROI 裁剪 x=%d y=%d w=%d h=%d", x, y, w, h)

    logger.info("预处理: CLAHE + 空间带通增强开始, frames=%d", work.shape[0])
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
        logger.info("预处理: 使用前 %d 帧中值作为参考帧", n_ref)
    else:
        reference_frame = enhanced[0].copy()
        logger.info("预处理: 使用首帧作为参考帧")

    warps = []
    aligned = []
    if dic_cfg.use_global_motion_correction:
        logger.info("预处理: 开始全局 ECC 刚体配准")
    for frame in enhanced:
        warp = _estimate_euclidean(reference_frame, frame) if dic_cfg.use_global_motion_correction else np.eye(2, 3, dtype=np.float32)
        warps.append(warp)
        aligned.append(_apply_warp(frame, warp))
    aligned = np.stack(aligned, axis=0)
    warps = np.stack(warps, axis=0)

    if dic_cfg.use_reference_region_correction:
        logger.info("预处理: 开始参考区运动估计, regions=%d", len(config.reference_regions))
        ref_motion = _region_mean_motion(aligned, config.reference_regions, roi_offset_xy)
    else:
        ref_motion = np.zeros((aligned.shape[0], 2), dtype=np.float32)

    common_mode = np.zeros(aligned.shape[0], dtype=np.float32)
    if dic_cfg.use_common_mode_subtraction:
        spatial_mean = aligned.mean(axis=(1, 2))
        common_mode = signal.detrend(spatial_mean, type="linear").astype(np.float32)
        aligned = aligned - common_mode[:, None, None]
        logger.info("预处理: 已执行公共模态扣除")

    return PreprocessResult(
        raw_frames=work,
        processed_frames=aligned,
        reference_frame=reference_frame,
        rigid_transforms=warps,
        reference_region_motion=ref_motion,
        common_mode_signal=common_mode,
        roi_offset_xy=roi_offset_xy,
        undistortion=undistortion,
    )
