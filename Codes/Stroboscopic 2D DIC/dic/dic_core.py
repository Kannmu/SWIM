from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage

from .config import ProjectConfig
from .types import DICResult, GridDefinition, PreprocessResult


def build_grid(image_shape: tuple[int, int], subset_size: int, step: int) -> GridDefinition:
    h, w = image_shape
    half = subset_size // 2
    xs = np.arange(half, w - half, step, dtype=np.int32)
    ys = np.arange(half, h - half, step, dtype=np.int32)
    return GridDefinition(centers_x=xs, centers_y=ys, subset_size_px=subset_size, step_size_px=step)


def _extract_patch(image: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    return image[cy - half:cy + half + 1, cx - half:cx + half + 1]


def _subpixel_parabola(values: np.ndarray) -> float:
    a, b, c = values
    denom = a - 2 * b + c
    if abs(denom) < 1e-6:
        return 0.0
    return 0.5 * (a - c) / denom


def _phase_refine(template: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    shift, _ = cv2.phaseCorrelate(template.astype(np.float32), target.astype(np.float32))
    return float(shift[0]), float(shift[1])


def run_dic(pre: PreprocessResult, config: ProjectConfig) -> DICResult:
    frames = pre.processed_frames
    dic_cfg = config.dic
    ref = pre.reference_frame
    grid = build_grid(ref.shape, dic_cfg.subset_size_px, dic_cfg.step_size_px)

    ny = len(grid.centers_y)
    nx = len(grid.centers_x)
    nt = frames.shape[0]
    u = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    v = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    corr = np.full((nt, ny, nx), np.nan, dtype=np.float32)

    half = dic_cfg.subset_size_px // 2
    search = dic_cfg.search_radius_px

    for t, frame in enumerate(frames):
        for iy, cy in enumerate(grid.centers_y):
            for ix, cx in enumerate(grid.centers_x):
                template = _extract_patch(ref, int(cx), int(cy), half)
                y0 = max(0, cy - half - search)
                y1 = min(frame.shape[0], cy + half + search + 1)
                x0 = max(0, cx - half - search)
                x1 = min(frame.shape[1], cx + half + search + 1)
                search_img = frame[y0:y1, x0:x1]
                if search_img.shape[0] < template.shape[0] or search_img.shape[1] < template.shape[1]:
                    continue
                res = cv2.matchTemplate(search_img.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                _, peak, _, peak_loc = cv2.minMaxLoc(res)
                px, py = peak_loc
                dx = px - (search_img.shape[1] - template.shape[1]) / 2.0
                dy = py - (search_img.shape[0] - template.shape[0]) / 2.0

                if 0 < py < res.shape[0] - 1:
                    dy += _subpixel_parabola(res[py - 1:py + 2, px])
                if 0 < px < res.shape[1] - 1:
                    dx += _subpixel_parabola(res[py, px - 1:px + 2])

                tx = int(np.clip(cx + dx, half, frame.shape[1] - half - 1))
                ty = int(np.clip(cy + dy, half, frame.shape[0] - half - 1))
                target = _extract_patch(frame, tx, ty, half)
                refine_x, refine_y = _phase_refine(template, target)
                dx += refine_x
                dy += refine_y

                if dic_cfg.use_reference_region_correction and t < len(pre.reference_region_motion):
                    dx -= float(pre.reference_region_motion[t, 0])
                    dy -= float(pre.reference_region_motion[t, 1])

                u[t, iy, ix] = dx
                v[t, iy, ix] = dy
                corr[t, iy, ix] = peak

    u = ndimage.median_filter(u, size=(1, 3, 3))
    v = ndimage.median_filter(v, size=(1, 3, 3))
    return DICResult(u=u, v=v, corr=corr, grid=grid)
