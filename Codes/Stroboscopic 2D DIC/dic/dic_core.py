from __future__ import annotations

import logging
import time

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from .acceleration import (
    AccelerationConfig,
    build_runtime,
    extract_patches,
    extract_patches_gpu,
    normalized_cc_batch,
    normalized_cc_batch_gpu,
    parabola_subpixel_1d,
    phase_refine_batch_cpu,
    phase_refine_batch_gpu,
    search_windows_batch,
    search_windows_batch_gpu,
)
from .config import ProjectConfig
from .types import DICResult, GridDefinition, PreprocessResult, RuntimeDiagnostics

logger = logging.getLogger("swim_dic")


def build_grid(image_shape: tuple[int, int], subset_size: int, step: int, search_radius: int = 0) -> GridDefinition:
    h, w = image_shape
    half = subset_size // 2
    margin = half + max(0, int(search_radius))
    xs = np.arange(margin, w - margin, step, dtype=np.int32)
    ys = np.arange(margin, h - margin, step, dtype=np.int32)
    return GridDefinition(centers_x=xs, centers_y=ys, subset_size_px=subset_size, step_size_px=step)


def _flatten_grid(grid: GridDefinition) -> tuple[np.ndarray, np.ndarray, int, int]:
    mesh_y, mesh_x = np.meshgrid(grid.centers_y, grid.centers_x, indexing="ij")
    return mesh_y.reshape(-1), mesh_x.reshape(-1), len(grid.centers_y), len(grid.centers_x)


def _target_centers(
    centers_y: np.ndarray,
    centers_x: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
    half: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    tx = np.clip(np.rint(centers_x.astype(np.float32) + dx), half, w - half - 1).astype(np.int32)
    ty = np.clip(np.rint(centers_y.astype(np.float32) + dy), half, h - half - 1).astype(np.int32)
    return ty, tx


def _reference_motion_array(pre: PreprocessResult, enabled: bool, t: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    if enabled and t < len(pre.reference_region_motion):
        dx = np.full(count, float(pre.reference_region_motion[t, 0]), dtype=np.float32)
        dy = np.full(count, float(pre.reference_region_motion[t, 1]), dtype=np.float32)
        return dx, dy
    zero = np.zeros(count, dtype=np.float32)
    return zero, zero


def _build_diagnostics(
    runtime_used_gpu: bool,
    runtime_backend: str,
    batch_size: int,
    nt: int,
    ny: int,
    nx: int,
    npoints: int,
    subset_size: int,
    search_radius: int,
) -> RuntimeDiagnostics:
    search_window_size_px = subset_size + 2 * search_radius
    estimated_positions = max(1, (2 * search_radius + 1) ** 2)
    return RuntimeDiagnostics(
        used_gpu=runtime_used_gpu,
        gpu_backend=runtime_backend,
        batch_size=batch_size,
        num_frames=nt,
        grid_shape=(ny, nx),
        num_points=npoints,
        total_matches=nt * npoints,
        search_window_size_px=search_window_size_px,
        subset_size_px=subset_size,
        search_radius_px=search_radius,
        estimated_search_positions_per_point=estimated_positions,
        estimated_fft_refinements=nt * npoints,
    )


def run_dic(pre: PreprocessResult, config: ProjectConfig) -> DICResult:
    frames = pre.processed_frames.astype(np.float32, copy=False)
    dic_cfg = config.dic
    ref = pre.reference_frame.astype(np.float32, copy=False)
    grid = build_grid(ref.shape, dic_cfg.subset_size_px, dic_cfg.step_size_px, dic_cfg.search_radius_px)

    flat_y, flat_x, ny, nx = _flatten_grid(grid)
    nt = frames.shape[0]
    npoints = flat_x.size
    u = np.full((nt, npoints), np.nan, dtype=np.float32)
    v = np.full((nt, npoints), np.nan, dtype=np.float32)
    corr = np.full((nt, npoints), np.nan, dtype=np.float32)

    half = dic_cfg.subset_size_px // 2
    search = dic_cfg.search_radius_px
    runtime = build_runtime(
        AccelerationConfig(
            enable_gpu=dic_cfg.enable_gpu,
            gpu_backend=dic_cfg.gpu_backend,
            gpu_batch_size=dic_cfg.gpu_batch_size,
            numba_parallel=dic_cfg.numba_parallel,
        )
    )
    diagnostics = _build_diagnostics(
        runtime_used_gpu=runtime.use_gpu,
        runtime_backend=runtime.gpu_backend,
        batch_size=runtime.batch_size,
        nt=nt,
        ny=ny,
        nx=nx,
        npoints=npoints,
        subset_size=dic_cfg.subset_size_px,
        search_radius=search,
    )

    logger.info("DIC: 启动局部相关计算")
    logger.info(
        "DIC: frames=%d, grid=%dx%d, points=%d, subset=%d, search_radius=%d, search_window=%d",
        diagnostics.num_frames,
        diagnostics.grid_shape[0],
        diagnostics.grid_shape[1],
        diagnostics.num_points,
        diagnostics.subset_size_px,
        diagnostics.search_radius_px,
        diagnostics.search_window_size_px,
    )
    logger.info(
        "DIC: total_matches=%d, estimated_search_positions_per_point=%d, estimated_fft_refinements=%d",
        diagnostics.total_matches,
        diagnostics.estimated_search_positions_per_point,
        diagnostics.estimated_fft_refinements,
    )
    logger.info(
        "DIC: acceleration backend=%s, used_gpu=%s, batch_size=%d, numba_parallel=%s",
        diagnostics.gpu_backend,
        diagnostics.used_gpu,
        diagnostics.batch_size,
        runtime.numba_parallel,
    )
    logger.info(
        "DIC: gpu_probe requested_backend=%s, gpu_available=%s, device_count=%d, device_name=%s",
        runtime.gpu_requested_backend,
        runtime.gpu_available,
        runtime.gpu_device_count,
        runtime.gpu_device_name or "unknown",
    )
    logger.info("DIC: gpu_probe reason=%s", runtime.gpu_reason)

    if runtime.use_gpu:
        templates_gpu = extract_patches_gpu(runtime.xp.asarray(ref, dtype=runtime.xp.float32), flat_y, flat_x, dic_cfg.subset_size_px)
    else:
        templates = extract_patches(ref, flat_y, flat_x, dic_cfg.subset_size_px)

    t_start_all = time.perf_counter()
    frame_iterator = tqdm(range(nt), desc="SWIM-DIC analyze", unit="frame")
    for t in frame_iterator:
        frame = frames[t]
        ref_dx, ref_dy = _reference_motion_array(pre, dic_cfg.use_reference_region_correction, t, npoints)
        frame_u = np.full(npoints, np.nan, dtype=np.float32)
        frame_v = np.full(npoints, np.nan, dtype=np.float32)
        frame_corr = np.full(npoints, np.nan, dtype=np.float32)
        frame_start = time.perf_counter()

        if runtime.use_gpu:
            frame_gpu = runtime.xp.asarray(frame, dtype=runtime.xp.float32)

        for start in range(0, npoints, runtime.batch_size):
            stop = min(start + runtime.batch_size, npoints)
            batch_y = flat_y[start:stop]
            batch_x = flat_x[start:stop]

            if runtime.use_gpu:
                batch_templates_gpu = templates_gpu[start:stop]
                search_windows_gpu = search_windows_batch_gpu(frame_gpu, batch_y, batch_x, dic_cfg.subset_size_px, search)
                response = normalized_cc_batch_gpu(search_windows_gpu, batch_templates_gpu)
                response_np = runtime.xp.asnumpy(response).astype(np.float32, copy=False)
            else:
                batch_templates = templates[start:stop]
                search_windows = search_windows_batch(frame, batch_y, batch_x, dic_cfg.subset_size_px, search)
                response = normalized_cc_batch(search_windows, batch_templates, runtime.xp)
                response_np = np.asarray(response, dtype=np.float32)

            batch_n = response_np.shape[0]
            flat_idx = np.argmax(response_np.reshape(batch_n, -1), axis=1)
            py, px = np.unravel_index(flat_idx, response_np.shape[1:])
            peak = response_np[np.arange(batch_n), py, px]

            max_y = response_np.shape[1] - 1
            max_x = response_np.shape[2] - 1
            dy = py.astype(np.float32) - (response_np.shape[1] - 1) / 2.0
            dx = px.astype(np.float32) - (response_np.shape[2] - 1) / 2.0

            valid_y = (py > 0) & (py < max_y)
            valid_x = (px > 0) & (px < max_x)
            if np.any(valid_y):
                idx = np.where(valid_y)[0]
                dy[idx] += parabola_subpixel_1d(
                    response_np[idx, py[idx] - 1, px[idx]],
                    response_np[idx, py[idx], px[idx]],
                    response_np[idx, py[idx] + 1, px[idx]],
                )
            if np.any(valid_x):
                idx = np.where(valid_x)[0]
                dx[idx] += parabola_subpixel_1d(
                    response_np[idx, py[idx], px[idx] - 1],
                    response_np[idx, py[idx], px[idx]],
                    response_np[idx, py[idx], px[idx] + 1],
                )

            target_y, target_x = _target_centers(batch_y, batch_x, dy, dx, half, frame.shape)

            if runtime.use_gpu:
                targets_gpu = extract_patches_gpu(frame_gpu, target_y, target_x, dic_cfg.subset_size_px)
                refine_gpu = phase_refine_batch_gpu(batch_templates_gpu, targets_gpu, upsample_factor=8)
                refine = runtime.xp.asnumpy(refine_gpu).astype(np.float32, copy=False)
            else:
                targets = extract_patches(frame, target_y, target_x, dic_cfg.subset_size_px)
                refine = phase_refine_batch_cpu(batch_templates, targets)

            dx += refine[:, 0]
            dy += refine[:, 1]
            dx -= ref_dx[start:stop]
            dy -= ref_dy[start:stop]

            frame_u[start:stop] = dx
            frame_v[start:stop] = dy
            frame_corr[start:stop] = peak

        u[t] = frame_u
        v[t] = frame_v
        corr[t] = frame_corr
        elapsed = time.perf_counter() - frame_start
        processed = t + 1
        total_elapsed = time.perf_counter() - t_start_all
        avg_per_frame = total_elapsed / processed
        eta = avg_per_frame * (nt - processed)
        frame_iterator.set_postfix_str(f"avg={avg_per_frame:.2f}s/frame eta={eta:.1f}s")
        logger.info(
            "DIC: frame %d/%d 完成, elapsed=%.2fs, cumulative=%.2fs, eta=%.2fs",
            processed,
            nt,
            elapsed,
            total_elapsed,
            eta,
        )

    logger.info("DIC: 开始 3x3 空间中值滤波")
    u = ndimage.median_filter(u.reshape(nt, ny, nx), size=(1, 3, 3))
    v = ndimage.median_filter(v.reshape(nt, ny, nx), size=(1, 3, 3))
    corr = corr.reshape(nt, ny, nx)
    logger.info("DIC: 全部完成, total_elapsed=%.2fs", time.perf_counter() - t_start_all)
    return DICResult(u=u, v=v, corr=corr, grid=grid, diagnostics=diagnostics)
