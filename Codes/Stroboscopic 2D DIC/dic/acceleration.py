from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from scipy import ndimage

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cupy_ndimage
except ImportError:
    cp = None
    cupy_ndimage = None

try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range

logger = logging.getLogger("swim_dic")


@dataclass(slots=True)
class AccelerationConfig:
    enable_gpu: bool = True
    gpu_backend: str = "auto"
    gpu_batch_size: int = 8
    numba_parallel: bool = True


@dataclass(slots=True)
class AccelerationRuntime:
    use_gpu: bool
    xp: object
    batch_size: int
    numba_available: bool
    numba_parallel: bool
    gpu_backend: str
    gpu_requested_backend: str = "auto"
    gpu_available: bool = False
    gpu_device_count: int = 0
    gpu_device_name: str | None = None
    gpu_reason: str = "GPU 未请求"
    debug_messages: list[str] = field(default_factory=list)


class _NumpyCompat:
    float32 = np.float32
    int32 = np.int32
    nan = np.nan
    newaxis = None

    @staticmethod
    def asarray(arr: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
        return np.asarray(arr, dtype=dtype)

    @staticmethod
    def asnumpy(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr)

    @staticmethod
    def arange(*args, **kwargs):
        return np.arange(*args, **kwargs)

    @staticmethod
    def sum(*args, **kwargs):
        return np.sum(*args, **kwargs)

    @staticmethod
    def sqrt(*args, **kwargs):
        return np.sqrt(*args, **kwargs)

    @staticmethod
    def maximum(*args, **kwargs):
        return np.maximum(*args, **kwargs)

    @staticmethod
    def clip(*args, **kwargs):
        return np.clip(*args, **kwargs)

    @staticmethod
    def argmax(*args, **kwargs):
        return np.argmax(*args, **kwargs)

    @staticmethod
    def unravel_index(*args, **kwargs):
        return np.unravel_index(*args, **kwargs)

    @staticmethod
    def stack(*args, **kwargs):
        return np.stack(*args, **kwargs)

    @staticmethod
    def nan_to_num(*args, **kwargs):
        return np.nan_to_num(*args, **kwargs)


np_compat = _NumpyCompat()


def build_runtime(config: AccelerationConfig) -> AccelerationRuntime:
    use_gpu = False
    xp = np_compat
    gpu_backend = "cpu"
    requested_backend = str(config.gpu_backend).lower()
    debug_messages: list[str] = []
    gpu_available = False
    gpu_device_count = 0
    gpu_device_name: str | None = None
    gpu_reason = "GPU 未请求"

    debug_messages.append(
        f"Acceleration: build_runtime(enable_gpu={config.enable_gpu}, gpu_backend={requested_backend}, gpu_batch_size={config.gpu_batch_size}, numba_parallel={config.numba_parallel})"
    )

    if not config.enable_gpu:
        gpu_reason = "配置 dic.enable_gpu=false，已强制使用 CPU"
        debug_messages.append(f"Acceleration: {gpu_reason}")
    elif requested_backend not in {"auto", "cupy"}:
        gpu_reason = f"不支持的 GPU backend={requested_backend}，已回退到 CPU"
        debug_messages.append(f"Acceleration: {gpu_reason}")
    elif cp is None:
        gpu_reason = "未安装 CuPy，无法启用 GPU"
        debug_messages.append("Acceleration: CuPy import failed, cp is None")
    else:
        try:
            gpu_device_count = int(cp.cuda.runtime.getDeviceCount())
            debug_messages.append(f"Acceleration: detected CUDA device count={gpu_device_count}")
            if gpu_device_count <= 0:
                gpu_reason = "CuPy 已安装，但未检测到 CUDA 设备"
                debug_messages.append(f"Acceleration: {gpu_reason}")
            else:
                device = cp.cuda.Device(0)
                with device:
                    props = cp.cuda.runtime.getDeviceProperties(device.id)
                name_raw = props.get("name", b"")
                if isinstance(name_raw, bytes):
                    gpu_device_name = name_raw.decode("utf-8", errors="ignore")
                else:
                    gpu_device_name = str(name_raw)
                gpu_available = True
                use_gpu = True
                xp = cp
                gpu_backend = "cupy"
                gpu_reason = f"已启用 CuPy GPU 加速，device[0]={gpu_device_name}"
                debug_messages.append(f"Acceleration: selected device[0]={gpu_device_name}")
                if cupy_ndimage is None:
                    debug_messages.append("Acceleration: warning: cupyx.scipy.ndimage is unavailable; GPU phase refinement may fail")
        except Exception as exc:
            use_gpu = False
            xp = np_compat
            gpu_backend = "cpu"
            gpu_reason = f"CUDA 运行时初始化失败，已回退到 CPU: {exc.__class__.__name__}: {exc}"
            debug_messages.append(f"Acceleration: runtime probe failed: {exc.__class__.__name__}: {exc}")

    logger.info(
        "Acceleration: requested_backend=%s, enable_gpu=%s, selected_backend=%s, use_gpu=%s, numba_available=%s, numba_parallel=%s",
        requested_backend,
        config.enable_gpu,
        gpu_backend,
        use_gpu,
        njit is not None,
        bool(config.numba_parallel),
    )
    if gpu_device_count > 0 or gpu_device_name:
        logger.info(
            "Acceleration: cuda_device_count=%d, device_name=%s",
            gpu_device_count,
            gpu_device_name or "unknown",
        )
    logger.info("Acceleration: %s", gpu_reason)
    for message in debug_messages:
        logger.debug(message)

    return AccelerationRuntime(
        use_gpu=use_gpu,
        xp=xp,
        batch_size=max(1, int(config.gpu_batch_size)),
        numba_available=njit is not None,
        numba_parallel=bool(config.numba_parallel),
        gpu_backend=gpu_backend,
        gpu_requested_backend=requested_backend,
        gpu_available=gpu_available,
        gpu_device_count=gpu_device_count,
        gpu_device_name=gpu_device_name,
        gpu_reason=gpu_reason,
        debug_messages=debug_messages,
    )


def to_numpy(arr: np.ndarray) -> np.ndarray:
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def extract_patches(image: np.ndarray, centers_y: np.ndarray, centers_x: np.ndarray, subset_size: int) -> np.ndarray:
    half = subset_size // 2
    windows = sliding_window_view(np.ascontiguousarray(image), (subset_size, subset_size))
    return windows[centers_y - half, centers_x - half].astype(np.float32, copy=False)


def extract_patches_gpu(image_gpu, centers_y: np.ndarray, centers_x: np.ndarray, subset_size: int):
    if cp is None:
        raise RuntimeError("CuPy 不可用，无法执行 GPU patch 提取")
    half = subset_size // 2
    windows = cp.lib.stride_tricks.sliding_window_view(cp.ascontiguousarray(image_gpu), (subset_size, subset_size))
    y_idx = cp.asarray(centers_y - half, dtype=cp.int32)
    x_idx = cp.asarray(centers_x - half, dtype=cp.int32)
    return windows[y_idx, x_idx].astype(cp.float32, copy=False)


def search_windows_batch(
    frame: np.ndarray,
    centers_y: np.ndarray,
    centers_x: np.ndarray,
    subset_size: int,
    search_radius: int,
) -> np.ndarray:
    frame = np.ascontiguousarray(frame, dtype=np.float32)
    win = subset_size + 2 * search_radius
    top = centers_y - subset_size // 2 - search_radius
    left = centers_x - subset_size // 2 - search_radius
    windows = sliding_window_view(frame, (win, win))
    return windows[top, left].astype(np.float32, copy=False)


def search_windows_batch_gpu(frame_gpu, centers_y: np.ndarray, centers_x: np.ndarray, subset_size: int, search_radius: int):
    if cp is None:
        raise RuntimeError("CuPy 不可用，无法执行 GPU search window 提取")
    frame_gpu = cp.ascontiguousarray(frame_gpu, dtype=cp.float32)
    win = subset_size + 2 * search_radius
    top = cp.asarray(centers_y - subset_size // 2 - search_radius, dtype=cp.int32)
    left = cp.asarray(centers_x - subset_size // 2 - search_radius, dtype=cp.int32)
    windows = cp.lib.stride_tricks.sliding_window_view(frame_gpu, (win, win))
    return windows[top, left].astype(cp.float32, copy=False)


def normalized_cc_batch(search_windows: np.ndarray, templates: np.ndarray, xp: object) -> np.ndarray:
    sw = xp.asarray(search_windows, dtype=xp.float32)
    tpl = xp.asarray(templates, dtype=xp.float32)
    tpl_h, tpl_w = tpl.shape[-2:]
    cand = sliding_window_view(sw, (tpl_h, tpl_w), axis=(1, 2))
    tpl_zm = tpl - xp.sum(tpl, axis=(1, 2), keepdims=True) / float(tpl_h * tpl_w)
    cand_mean = xp.sum(cand, axis=(-1, -2), keepdims=True) / float(tpl_h * tpl_w)
    cand_zm = cand - cand_mean
    numer = xp.sum(cand_zm * tpl_zm[:, xp.newaxis, xp.newaxis, :, :], axis=(-1, -2))
    cand_energy = xp.sum(cand_zm * cand_zm, axis=(-1, -2))
    tpl_energy = xp.sum(tpl_zm * tpl_zm, axis=(-1, -2))[:, xp.newaxis, xp.newaxis]
    denom = xp.sqrt(xp.maximum(cand_energy * tpl_energy, 1e-12))
    return numer / denom


def normalized_cc_batch_gpu(search_windows_gpu, templates_gpu):
    if cp is None:
        raise RuntimeError("CuPy 不可用，无法执行 GPU NCC")
    sw = cp.asarray(search_windows_gpu, dtype=cp.float32)
    tpl = cp.asarray(templates_gpu, dtype=cp.float32)
    tpl_h, tpl_w = tpl.shape[-2:]
    cand = cp.lib.stride_tricks.sliding_window_view(sw, (tpl_h, tpl_w), axis=(1, 2))
    tpl_zm = tpl - cp.mean(tpl, axis=(1, 2), keepdims=True)
    cand_mean = cp.mean(cand, axis=(-1, -2), keepdims=True)
    cand_zm = cand - cand_mean
    numer = cp.sum(cand_zm * tpl_zm[:, cp.newaxis, cp.newaxis, :, :], axis=(-1, -2))
    cand_energy = cp.sum(cand_zm * cand_zm, axis=(-1, -2))
    tpl_energy = cp.sum(tpl_zm * tpl_zm, axis=(-1, -2))[:, cp.newaxis, cp.newaxis]
    denom = cp.sqrt(cp.maximum(cand_energy * tpl_energy, 1e-12))
    return numer / denom


def parabola_subpixel_1d(left: np.ndarray, center: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = left - 2.0 * center + right
    out = np.zeros_like(center, dtype=np.float32)
    mask = np.abs(denom) >= 1e-6
    out[mask] = 0.5 * (left[mask] - right[mask]) / denom[mask]
    return out


def phase_refine_batch_cpu(templates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if templates.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if njit is None:
        return _phase_refine_batch_cpu_numpy(templates, targets)
    return _phase_refine_batch_numba(templates, targets)


def phase_refine_batch_gpu(templates_gpu, targets_gpu, upsample_factor: int = 8):
    if cp is None or cupy_ndimage is None:
        raise RuntimeError("CuPy/cupyx 不可用，无法执行 GPU 相位细化")
    if templates_gpu.size == 0:
        return cp.zeros((0, 2), dtype=cp.float32)
    n = int(templates_gpu.shape[0])
    out = cp.zeros((n, 2), dtype=cp.float32)
    for i in range(n):
        out[i] = _phase_refine_single_gpu(templates_gpu[i], targets_gpu[i], upsample_factor=upsample_factor)
    return out


def _phase_refine_batch_cpu_numpy(templates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out = np.zeros((templates.shape[0], 2), dtype=np.float32)
    for i in range(templates.shape[0]):
        out[i] = _phase_refine_single_numpy(templates[i], targets[i])
    return out


def _phase_refine_single_numpy(template: np.ndarray, target: np.ndarray, upsample_factor: int = 8) -> np.ndarray:
    f1 = np.fft.fft2(template.astype(np.float32))
    f2 = np.fft.fft2(target.astype(np.float32))
    cps = f1 * np.conj(f2)
    mag = np.abs(cps)
    cps /= np.where(mag < 1e-12, 1.0, mag)
    corr = np.fft.ifft2(cps)
    corr_abs = np.abs(corr)
    peak = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    py = int(peak[0])
    px = int(peak[1])
    h, w = corr_abs.shape
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w
    refined = _upsampled_peak_offset_numpy(corr_abs, py, px, upsample_factor)
    return np.array([refined[0], refined[1]], dtype=np.float32)


def _upsampled_peak_offset_numpy(corr_abs: np.ndarray, py: int, px: int, upsample_factor: int) -> tuple[float, float]:
    h, w = corr_abs.shape
    peak_y = py % h
    peak_x = px % w
    y0 = max(peak_y - 1, 0)
    y1 = min(peak_y + 2, h)
    x0 = max(peak_x - 1, 0)
    x1 = min(peak_x + 2, w)
    patch = corr_abs[y0:y1, x0:x1]
    if patch.size == 0:
        return float(px), float(py)
    zoomed = ndimage.zoom(patch, upsample_factor, order=3)
    if zoomed.size == 0:
        return float(px), float(py)
    peak_zoom = np.unravel_index(np.argmax(zoomed), zoomed.shape)
    sub_y = y0 + peak_zoom[0] / upsample_factor
    sub_x = x0 + peak_zoom[1] / upsample_factor
    if sub_y > h / 2:
        sub_y -= h
    if sub_x > w / 2:
        sub_x -= w
    return float(sub_x), float(sub_y)


def _phase_refine_single_gpu(template_gpu, target_gpu, upsample_factor: int = 8):
    f1 = cp.fft.fft2(template_gpu.astype(cp.float32))
    f2 = cp.fft.fft2(target_gpu.astype(cp.float32))
    cps = f1 * cp.conj(f2)
    mag = cp.abs(cps)
    cps = cps / cp.where(mag < 1e-12, 1.0, mag)
    corr = cp.fft.ifft2(cps)
    corr_abs = cp.abs(corr)
    peak = cp.unravel_index(cp.argmax(corr_abs), corr_abs.shape)
    peak_y = int(peak[0].item())
    peak_x = int(peak[1].item())
    h, w = corr_abs.shape
    py = peak_y
    px = peak_x
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w
    y0 = max(peak_y - 1, 0)
    y1 = min(peak_y + 2, h)
    x0 = max(peak_x - 1, 0)
    x1 = min(peak_x + 2, w)
    patch = corr_abs[y0:y1, x0:x1]
    if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
        return cp.asarray([float(px), float(py)], dtype=cp.float32)
    zoom_order = 3 if patch.shape[0] > 3 and patch.shape[1] > 3 else 1
    zoomed = cupy_ndimage.zoom(patch, upsample_factor, order=zoom_order)
    if zoomed.size == 0 or zoomed.shape[0] == 0 or zoomed.shape[1] == 0:
        return cp.asarray([float(px), float(py)], dtype=cp.float32)
    peak_zoom = cp.unravel_index(cp.argmax(zoomed), zoomed.shape)
    sub_y = y0 + float(peak_zoom[0].item()) / upsample_factor
    sub_x = x0 + float(peak_zoom[1].item()) / upsample_factor
    if sub_y > h / 2:
        sub_y -= h
    if sub_x > w / 2:
        sub_x -= w
    return cp.asarray([sub_x, sub_y], dtype=cp.float32)


if njit is not None:
    @njit(cache=True, fastmath=False)
    def _complex_abs_numba(value: complex) -> float:
        return math.sqrt(value.real * value.real + value.imag * value.imag)


    @njit(cache=True, fastmath=False)
    def _phase_refine_single_numba(template: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        f1 = np.fft.fft2(template)
        f2 = np.fft.fft2(target)
        cps = f1 * np.conjugate(f2)
        h, w = cps.shape
        for y in range(h):
            for x in range(w):
                mag = _complex_abs_numba(cps[y, x])
                if mag >= 1e-12:
                    cps[y, x] = cps[y, x] / mag
                else:
                    cps[y, x] = 0.0 + 0.0j
        corr = np.fft.ifft2(cps)
        best_y = 0
        best_x = 0
        best_val = -1.0
        for y in range(h):
            for x in range(w):
                mag = _complex_abs_numba(corr[y, x])
                if mag > best_val:
                    best_val = mag
                    best_y = y
                    best_x = x
        if best_y > h // 2:
            best_y -= h
        if best_x > w // 2:
            best_x -= w
        return float(best_x), float(best_y)


    @njit(cache=True, parallel=True, fastmath=False)
    def _phase_refine_batch_numba(templates: np.ndarray, targets: np.ndarray) -> np.ndarray:
        n = templates.shape[0]
        out = np.zeros((n, 2), dtype=np.float32)
        for i in prange(n):
            dx, dy = _phase_refine_single_numba(templates[i], targets[i])
            out[i, 0] = dx
            out[i, 1] = dy
        return out
