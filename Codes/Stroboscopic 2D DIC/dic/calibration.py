from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .config import ProjectConfig, load_config
from .io_utils import ensure_parent


@dataclass(slots=True)
class CalibrationBoard:
    inner_corners_rows: int
    inner_corners_cols: int
    square_size_mm: float


@dataclass(slots=True)
class CalibrationImageResult:
    image_path: str
    detected: bool
    reprojection_error_px: float | None
    pixels_per_square_x: float | None
    pixels_per_square_y: float | None


@dataclass(slots=True)
class CalibrationResult:
    image_size: tuple[int, int]
    image_count: int
    valid_image_count: int
    board: CalibrationBoard
    rms_reprojection_error_px: float
    mean_reprojection_error_px: float
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    optimal_camera_matrix: list[list[float]]
    roi: list[int]
    mean_pixel_size_um: float
    pixel_size_std_um: float
    mean_pixels_per_mm_x: float
    mean_pixels_per_mm_y: float
    image_results: list[CalibrationImageResult]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_size"] = list(self.image_size)
        return payload


@dataclass(slots=True)
class CalibrationFrameAssessment:
    detected: bool
    score: float
    message: str
    corners: np.ndarray | None
    coverage_x: float
    coverage_y: float
    sharpness: float
    exposure_score: float
    mean_intensity: float
    saturation_fraction: float


@dataclass(slots=True)
class CalibrationCaptureState:
    latest_frame: np.ndarray | None
    latest_assessment: CalibrationFrameAssessment
    last_detection_frame: np.ndarray | None
    last_detection_assessment: CalibrationFrameAssessment
    saved_paths: list[Path]
    next_index: int
    capture_requested: bool
    frame_index: int
    detection_frame_skip: int
    preview_max_width: int
    display_scale: float
    last_loop_timestamp: float
    fps_ema: float


class CalibrationError(RuntimeError):
    pass


_CAPTURE_WINDOW_NAME = "SWIM Calibration Capture"
_CAPTURE_BUTTON_NAME = "Capture best frame"
_CAPTURE_MIN_SCORE_DEFAULT = 70.0
_CAPTURE_RESIZE_WIDTH = 1280
_CAPTURE_BUTTON_RECT = (20, 244, 320, 56)
_CAPTURE_STATUS_RECT = (12, 12, 1120, 220)
_CAPTURE_DEFAULT_PREVIEW_MAX_WIDTH = 1280
_CAPTURE_MAX_PROCESS_WIDTH = 1280
_CAPTURE_DETECTION_FRAME_SKIP = 2
_CAPTURE_MIN_UPDATE_INTERVAL_S = 0.001
_CAPTURE_LOW_LIGHT_MEAN_THRESHOLD = 25.0
_CAPTURE_SATURATION_LOW_THRESHOLD = 3.0
_CAPTURE_SATURATION_HIGH_THRESHOLD = 252.0
_CAPTURE_TARGET_MEAN_INTENSITY = 145.0


def _build_object_points(board: CalibrationBoard) -> np.ndarray:
    grid = np.zeros((board.inner_corners_rows * board.inner_corners_cols, 3), dtype=np.float32)
    xs, ys = np.meshgrid(
        np.arange(board.inner_corners_cols, dtype=np.float32),
        np.arange(board.inner_corners_rows, dtype=np.float32),
    )
    grid[:, 0] = xs.reshape(-1) * board.square_size_mm
    grid[:, 1] = ys.reshape(-1) * board.square_size_mm
    return grid


def _collect_image_paths(images_dir: Path) -> list[Path]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(images_dir.glob(pattern))
    return sorted(set(paths))


def _resize_by_max_width(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    if max_width <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    if width <= max_width:
        return image, 1.0
    scale = max_width / float(width)
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _prepare_grayscale_for_detection(frame: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    return _resize_by_max_width(gray, _CAPTURE_MAX_PROCESS_WIDTH)


def _find_corners(gray: np.ndarray, board: CalibrationBoard) -> np.ndarray | None:
    pattern_size = (board.inner_corners_cols, board.inner_corners_rows)
    detector = getattr(cv2, "findChessboardCornersSB", None)
    if callable(detector):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, corners = detector(gray, pattern_size, sb_flags)
        if found:
            return corners.reshape(-1, 2)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-4,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2)


def _estimate_pixel_size_um(corners: np.ndarray, board: CalibrationBoard) -> tuple[float, float, float, float]:
    grid = corners.reshape(board.inner_corners_rows, board.inner_corners_cols, 2)
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    mean_px_x = float(np.mean(dx)) if dx.size else float("nan")
    mean_px_y = float(np.mean(dy)) if dy.size else float("nan")
    pixels_per_mm_x = mean_px_x / board.square_size_mm
    pixels_per_mm_y = mean_px_y / board.square_size_mm
    pixel_size_um_x = 1000.0 / pixels_per_mm_x
    pixel_size_um_y = 1000.0 / pixels_per_mm_y
    return pixel_size_um_x, pixel_size_um_y, pixels_per_mm_x, pixels_per_mm_y


def _per_image_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    projected = projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((image_points - projected) ** 2, axis=1))))


def _normalize_score(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def assess_calibration_frame(frame: np.ndarray, board: CalibrationBoard) -> CalibrationFrameAssessment:
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    gray, scale = _prepare_grayscale_for_detection(frame)
    corners = _find_corners(gray, board)

    fallback_mean_intensity = float(np.mean(gray_full)) if gray_full.size else 0.0
    fallback_saturation_fraction = float(
        np.mean(
            (gray_full <= _CAPTURE_SATURATION_LOW_THRESHOLD)
            | (gray_full >= _CAPTURE_SATURATION_HIGH_THRESHOLD)
        )
    ) if gray_full.size else 0.0

    if corners is None:
        message = "No complete chessboard pattern detected"
        if fallback_mean_intensity < _CAPTURE_LOW_LIGHT_MEAN_THRESHOLD:
            message = "Image is too dark for chessboard detection"
        return CalibrationFrameAssessment(
            detected=False,
            score=0.0,
            message=message,
            corners=None,
            coverage_x=0.0,
            coverage_y=0.0,
            sharpness=float(cv2.Laplacian(gray_full, cv2.CV_64F).var()) if gray_full.size else 0.0,
            exposure_score=0.0,
            mean_intensity=fallback_mean_intensity,
            saturation_fraction=fallback_saturation_fraction,
        )

    if scale != 1.0:
        corners = corners / scale

    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)
    board_width = float(max_xy[0] - min_xy[0])
    board_height = float(max_xy[1] - min_xy[1])
    image_height, image_width = gray_full.shape[:2]
    coverage_x = board_width / float(max(image_width, 1))
    coverage_y = board_height / float(max(image_height, 1))

    hull = cv2.convexHull(corners.astype(np.float32))
    mask = np.zeros_like(gray_full, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    roi_pixels = gray_full[mask > 0]
    if roi_pixels.size == 0:
        roi_pixels = gray_full.reshape(-1)

    x0 = max(int(np.floor(min_xy[0])) - 12, 0)
    y0 = max(int(np.floor(min_xy[1])) - 12, 0)
    x1 = min(int(np.ceil(max_xy[0])) + 12, image_width)
    y1 = min(int(np.ceil(max_xy[1])) + 12, image_height)
    board_patch = gray_full[y0:y1, x0:x1]
    if board_patch.size == 0:
        board_patch = gray_full

    sharpness = float(cv2.Laplacian(board_patch, cv2.CV_64F).var()) if board_patch.size else 0.0
    mean_intensity = float(np.mean(roi_pixels)) if roi_pixels.size > 0 else fallback_mean_intensity
    saturation_fraction = float(
        np.mean(
            (roi_pixels <= _CAPTURE_SATURATION_LOW_THRESHOLD)
            | (roi_pixels >= _CAPTURE_SATURATION_HIGH_THRESHOLD)
        )
    ) if roi_pixels.size > 0 else fallback_saturation_fraction

    coverage_score = 0.5 * (
        _normalize_score(coverage_x, 0.10, 0.55) + _normalize_score(coverage_y, 0.10, 0.55)
    )
    sharpness_score = _normalize_score(sharpness, 30.0, 320.0)
    exposure_center_score = 1.0 - min(
        abs(mean_intensity - _CAPTURE_TARGET_MEAN_INTENSITY) / _CAPTURE_TARGET_MEAN_INTENSITY,
        1.0,
    )
    saturation_score = 1.0 - min(saturation_fraction / 0.05, 1.0)
    exposure_score = float(np.clip(0.75 * exposure_center_score + 0.25 * saturation_score, 0.0, 1.0))

    final_score = float(
        np.clip(
            100.0 * (0.55 * coverage_score + 0.20 * sharpness_score + 0.25 * exposure_score),
            0.0,
            100.0,
        )
    )

    if coverage_score < 0.20:
        message = "Pattern detected but too small—move closer or let it span more of the frame"
    elif sharpness_score < 0.20:
        message = "Pattern detected but blurry—hold still or refocus"
    elif exposure_score < 0.25:
        message = "Pattern detected but exposure is poor—adjust lighting or camera exposure"
    elif final_score >= 85.0:
        message = "Excellent image quality—ready to capture"
    elif final_score >= 70.0:
        message = "Good image quality—capture allowed"
    else:
        message = "Pattern detected—improve angle, coverage, or sharpness for a better sample"

    return CalibrationFrameAssessment(
        detected=True,
        score=final_score,
        message=message,
        corners=corners,
        coverage_x=coverage_x,
        coverage_y=coverage_y,
        sharpness=sharpness,
        exposure_score=100.0 * exposure_score,
        mean_intensity=mean_intensity,
        saturation_fraction=saturation_fraction,
    )


def _format_capture_filename(index: int) -> str:
    return f"calib_{index:04d}.png"


def _next_capture_index(output_dir: Path) -> int:
    existing = sorted(output_dir.glob("calib_*.png"))
    indices: list[int] = []
    for path in existing:
        stem = path.stem
        suffix = stem.split("_")[-1]
        if suffix.isdigit():
            indices.append(int(suffix))
    return (max(indices) + 1) if indices else 1


def _open_camera(camera_index: int, backend: int | None) -> cv2.VideoCapture:
    capture_backend = backend if backend is not None else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, capture_backend)
    if not cap.isOpened():
        raise CalibrationError(f"Failed to open camera index {camera_index}")
    return cap


def _configure_camera(cap: cv2.VideoCapture, config: ProjectConfig) -> None:
    camera_cfg = config.camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
    cap.set(cv2.CAP_PROP_FPS, camera_cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if camera_cfg.exposure_us is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(camera_cfg.exposure_us))
    if camera_cfg.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(camera_cfg.gain))


def _draw_multiline_text(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    font_scale: float,
    line_step: int,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        y += line_step


def _draw_capture_overlay(
    frame: np.ndarray,
    board: CalibrationBoard,
    assessment: CalibrationFrameAssessment,
    saved_count: int,
    min_score: float,
    preview_scale: float,
    fps_value: float,
) -> np.ndarray:
    display = frame.copy()

    if assessment.corners is not None:
        pattern_size = (board.inner_corners_cols, board.inner_corners_rows)
        draw_corners = (assessment.corners * preview_scale).reshape(-1, 1, 2)
        cv2.drawChessboardCorners(display, pattern_size, draw_corners, True)

    panel_x, panel_y, panel_w, panel_h = _CAPTURE_STATUS_RECT
    panel_right = min(panel_x + panel_w, display.shape[1] - 12)
    panel_bottom = min(panel_y + panel_h, display.shape[0] - 12)
    overlay = display.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_right, panel_bottom), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, display, 0.55, 0.0, display)
    cv2.rectangle(display, (panel_x, panel_y), (panel_right, panel_bottom), (180, 180, 180), 1)

    score_ready = assessment.detected and assessment.score >= min_score
    text_color = (0, 220, 0) if score_ready else (0, 200, 255)
    primary_color = text_color if assessment.detected else (0, 0, 255)

    lines = [
        f"Saved: {saved_count}    Live FPS: {fps_value:.1f}",
        f"Score: {assessment.score:.1f} / 100    Threshold: {min_score:.1f}",
        f"Coverage: {assessment.coverage_x * 100:.1f}% x {assessment.coverage_y * 100:.1f}%    Sharpness: {assessment.sharpness:.1f}",
        f"Exposure: {assessment.exposure_score:.1f}    Mean gray: {assessment.mean_intensity:.1f}    Saturation: {assessment.saturation_fraction * 100:.1f}%",
        assessment.message,
        "Click button or press SPACE to capture. Press Q or ESC to quit.",
    ]
    _draw_multiline_text(display, lines, (24, 42), 0.7, 32, primary_color)

    left, top, width, height = _CAPTURE_BUTTON_RECT
    button_left = left
    button_top = top
    button_right = min(button_left + width, display.shape[1] - 12)
    button_bottom = min(button_top + height, display.shape[0] - 12)
    button_color = (0, 170, 0) if score_ready else (100, 100, 100)
    cv2.rectangle(display, (button_left, button_top), (button_right, button_bottom), (0, 0, 0), -1)
    cv2.rectangle(display, (button_left, button_top), (button_right, button_bottom), button_color, 2)
    cv2.putText(
        display,
        _CAPTURE_BUTTON_NAME,
        (button_left + 12, button_top + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        button_color,
        2,
        cv2.LINE_AA,
    )
    return display


def _empty_assessment(message: str) -> CalibrationFrameAssessment:
    return CalibrationFrameAssessment(
        detected=False,
        score=0.0,
        message=message,
        corners=None,
        coverage_x=0.0,
        coverage_y=0.0,
        sharpness=0.0,
        exposure_score=0.0,
        mean_intensity=0.0,
        saturation_fraction=0.0,
    )


def _update_fps_ema(previous_timestamp: float, previous_fps_ema: float) -> tuple[float, float]:
    now = time.perf_counter()
    delta = max(now - previous_timestamp, _CAPTURE_MIN_UPDATE_INTERVAL_S)
    inst_fps = 1.0 / delta
    fps_ema = inst_fps if previous_fps_ema <= 0.0 else 0.85 * previous_fps_ema + 0.15 * inst_fps
    return now, fps_ema


def collect_calibration_images_interactive(
    config_path: str | Path,
    output_dir: str | Path,
    inner_corners_rows: int,
    inner_corners_cols: int,
    square_size_mm: float,
    min_score: float = _CAPTURE_MIN_SCORE_DEFAULT,
) -> list[Path]:
    config = load_config(config_path)
    board = CalibrationBoard(
        inner_corners_rows=inner_corners_rows,
        inner_corners_cols=inner_corners_cols,
        square_size_mm=square_size_mm,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = _open_camera(config.camera.camera_index, config.camera.backend)
    _configure_camera(cap, config)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width <= 0:
        actual_width = int(config.camera.width)
    if actual_height <= 0:
        actual_height = int(config.camera.height)

    preview_max_width = min(max(actual_width, 640), _CAPTURE_DEFAULT_PREVIEW_MAX_WIDTH)
    detection_frame_skip = max(1, _CAPTURE_DETECTION_FRAME_SKIP)
    if actual_width * actual_height >= 1920 * 1080:
        detection_frame_skip = max(detection_frame_skip, 3)

    state = CalibrationCaptureState(
        latest_frame=None,
        latest_assessment=_empty_assessment("Waiting for camera feed..."),
        last_detection_frame=None,
        last_detection_assessment=_empty_assessment("Waiting for first detection..."),
        saved_paths=[],
        next_index=_next_capture_index(output_path),
        capture_requested=False,
        frame_index=0,
        detection_frame_skip=detection_frame_skip,
        preview_max_width=preview_max_width,
        display_scale=1.0,
        last_loop_timestamp=time.perf_counter(),
        fps_ema=0.0,
    )

    def on_mouse(event: int, x: int, y: int, *_: Any) -> None:
        nonlocal state
        if event != cv2.EVENT_LBUTTONUP:
            return
        left, top, width, height = _CAPTURE_BUTTON_RECT
        if left <= x <= left + width and top <= y <= top + height:
            state.capture_requested = True

    cv2.namedWindow(_CAPTURE_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(_CAPTURE_WINDOW_NAME, min(_CAPTURE_RESIZE_WIDTH, preview_max_width), 900)
    cv2.setMouseCallback(_CAPTURE_WINDOW_NAME, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise CalibrationError("读取摄像头画面失败")

            state.frame_index += 1
            state.latest_frame = frame.copy()
            state.last_loop_timestamp, state.fps_ema = _update_fps_ema(state.last_loop_timestamp, state.fps_ema)

            should_run_detection = (
                state.last_detection_frame is None
                or state.frame_index % state.detection_frame_skip == 1
                or state.capture_requested
            )
            if should_run_detection:
                state.last_detection_frame = state.latest_frame.copy()
                state.last_detection_assessment = assess_calibration_frame(state.last_detection_frame, board)

            state.latest_assessment = state.last_detection_assessment
            display, preview_scale = _resize_by_max_width(state.latest_frame, state.preview_max_width)
            state.display_scale = preview_scale
            display = _draw_capture_overlay(
                frame=display,
                board=board,
                assessment=state.latest_assessment,
                saved_count=len(state.saved_paths),
                min_score=min_score,
                preview_scale=preview_scale,
                fps_value=state.fps_ema,
            )
            cv2.imshow(_CAPTURE_WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:
                state.capture_requested = True

            if not state.capture_requested or state.latest_frame is None:
                continue

            assessment_for_capture = assess_calibration_frame(state.latest_frame, board)
            state.latest_assessment = assessment_for_capture
            state.last_detection_frame = state.latest_frame.copy()
            state.last_detection_assessment = assessment_for_capture
            state.capture_requested = False

            if not assessment_for_capture.detected:
                print(f"跳过保存：{assessment_for_capture.message}")
                continue
            if assessment_for_capture.score < min_score:
                print(
                    "跳过保存：当前画面评分不足 "
                    f"({assessment_for_capture.score:.1f} < {min_score:.1f})"
                )
                continue

            file_path = output_path / _format_capture_filename(state.next_index)
            success = cv2.imwrite(str(file_path), state.latest_frame)
            if not success:
                raise CalibrationError(f"无法写入标定图像: {file_path}")
            state.saved_paths.append(file_path)
            state.next_index += 1
            print(f"已保存标定图像: {file_path}，评分 {assessment_for_capture.score:.1f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return state.saved_paths


def calibrate_camera_from_images(
    images_dir: str | Path,
    board: CalibrationBoard,
) -> CalibrationResult:
    images_path = Path(images_dir)
    image_paths = _collect_image_paths(images_path)
    if not image_paths:
        raise CalibrationError(f"标定目录中未找到图像: {images_path}")

    object_template = _build_object_points(board)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_results: list[CalibrationImageResult] = []
    image_size: tuple[int, int] | None = None
    pixel_sizes_um: list[float] = []
    pixels_per_mm_x_values: list[float] = []
    pixels_per_mm_y_values: list[float] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image_results.append(
                CalibrationImageResult(
                    image_path=str(image_path),
                    detected=False,
                    reprojection_error_px=None,
                    pixels_per_square_x=None,
                    pixels_per_square_y=None,
                )
            )
            continue

        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        elif image_size != (image.shape[1], image.shape[0]):
            raise CalibrationError("所有标定图像的分辨率必须一致")

        corners = _find_corners(image, board)
        if corners is None:
            image_results.append(
                CalibrationImageResult(
                    image_path=str(image_path),
                    detected=False,
                    reprojection_error_px=None,
                    pixels_per_square_x=None,
                    pixels_per_square_y=None,
                )
            )
            continue

        px_um_x, px_um_y, ppm_x, ppm_y = _estimate_pixel_size_um(corners, board)
        pixel_sizes_um.append(float(0.5 * (px_um_x + px_um_y)))
        pixels_per_mm_x_values.append(ppm_x)
        pixels_per_mm_y_values.append(ppm_y)
        object_points.append(object_template.copy())
        image_points.append(corners.astype(np.float32))
        image_results.append(
            CalibrationImageResult(
                image_path=str(image_path),
                detected=True,
                reprojection_error_px=None,
                pixels_per_square_x=ppm_x * board.square_size_mm,
                pixels_per_square_y=ppm_y * board.square_size_mm,
            )
        )

    if len(image_points) < 3:
        raise CalibrationError("至少需要 3 张成功检测到棋盘格的图像才能完成标定")
    assert image_size is not None

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    optimal_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, image_size, 1.0, image_size)

    reprojection_errors: list[float] = []
    valid_idx = 0
    for item in image_results:
        if not item.detected:
            continue
        err = _per_image_reprojection_error(
            object_points[valid_idx],
            image_points[valid_idx],
            rvecs[valid_idx],
            tvecs[valid_idx],
            camera_matrix,
            distortion,
        )
        item.reprojection_error_px = err
        reprojection_errors.append(err)
        valid_idx += 1

    return CalibrationResult(
        image_size=image_size,
        image_count=len(image_paths),
        valid_image_count=len(image_points),
        board=board,
        rms_reprojection_error_px=float(rms),
        mean_reprojection_error_px=float(np.mean(reprojection_errors)),
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=distortion.reshape(-1).tolist(),
        optimal_camera_matrix=optimal_matrix.tolist(),
        roi=[int(v) for v in roi],
        mean_pixel_size_um=float(np.mean(pixel_sizes_um)),
        pixel_size_std_um=float(np.std(pixel_sizes_um, ddof=0)),
        mean_pixels_per_mm_x=float(np.mean(pixels_per_mm_x_values)),
        mean_pixels_per_mm_y=float(np.mean(pixels_per_mm_y_values)),
        image_results=image_results,
    )


def save_calibration_result(result: CalibrationResult, path: str | Path) -> Path:
    target = Path(path)
    ensure_parent(target)
    target.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def update_config_with_calibration(config_path: str | Path, result: CalibrationResult) -> Path:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload.setdefault("analysis", {})
    payload.setdefault("calibration", {})
    payload["analysis"]["pixel_size_um"] = round(result.mean_pixel_size_um, 6)
    payload["calibration"]["enabled"] = True
    payload["calibration"]["board"] = {
        "inner_corners_rows": result.board.inner_corners_rows,
        "inner_corners_cols": result.board.inner_corners_cols,
        "square_size_mm": result.board.square_size_mm,
    }
    payload["calibration"]["camera_matrix"] = result.camera_matrix
    payload["calibration"]["distortion_coefficients"] = result.distortion_coefficients
    payload["calibration"]["optimal_camera_matrix"] = result.optimal_camera_matrix
    payload["calibration"]["roi"] = result.roi
    payload["calibration"]["image_size"] = list(result.image_size)
    payload["calibration"]["mean_reprojection_error_px"] = round(result.mean_reprojection_error_px, 6)
    payload["calibration"]["rms_reprojection_error_px"] = round(result.rms_reprojection_error_px, 6)
    payload["calibration"]["pixel_size_um"] = round(result.mean_pixel_size_um, 6)
    payload["calibration"]["pixel_size_std_um"] = round(result.pixel_size_std_um, 6)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def calibrate_from_config(
    config_path: str | Path,
    images_dir: str | Path,
    inner_corners_rows: int,
    inner_corners_cols: int,
    square_size_mm: float,
    output_json: str | Path | None = None,
    write_config: bool = True,
) -> CalibrationResult:
    load_config(config_path)
    board = CalibrationBoard(
        inner_corners_rows=inner_corners_rows,
        inner_corners_cols=inner_corners_cols,
        square_size_mm=square_size_mm,
    )
    result = calibrate_camera_from_images(images_dir=images_dir, board=board)
    if output_json is not None:
        save_calibration_result(result, output_json)
    if write_config:
        update_config_with_calibration(config_path, result)
    return result


def capture_and_calibrate_from_config(
    config_path: str | Path,
    images_dir: str | Path,
    inner_corners_rows: int,
    inner_corners_cols: int,
    square_size_mm: float,
    output_json: str | Path | None = None,
    write_config: bool = True,
    min_score: float = _CAPTURE_MIN_SCORE_DEFAULT,
) -> tuple[list[Path], CalibrationResult]:
    saved_paths = collect_calibration_images_interactive(
        config_path=config_path,
        output_dir=images_dir,
        inner_corners_rows=inner_corners_rows,
        inner_corners_cols=inner_corners_cols,
        square_size_mm=square_size_mm,
        min_score=min_score,
    )
    result = calibrate_from_config(
        config_path=config_path,
        images_dir=images_dir,
        inner_corners_rows=inner_corners_rows,
        inner_corners_cols=inner_corners_cols,
        square_size_mm=square_size_mm,
        output_json=output_json,
        write_config=write_config,
    )
    return saved_paths, result
