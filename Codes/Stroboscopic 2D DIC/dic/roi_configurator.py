from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .config import ProjectConfig, ReferenceRegion, load_config
from .io_utils import load_video_gray


_CONFIG_WINDOW_NAME = "SWIM ROI Configurator"
_PANEL_WIDTH = 420
_MAX_PREVIEW_WIDTH = 1400
_STATUS_HEIGHT = 190
_TEXT_COLOR = (235, 235, 235)
_PANEL_BG = (26, 26, 26)
_PANEL_ACCENT = (55, 145, 255)
_ROI_COLOR = (0, 220, 0)
_REF_COLOR = (255, 180, 0)
_ACTIVE_COLOR = (0, 90, 255)
_HOVER_COLOR = (255, 255, 255)
_HANDLE_HALF_SIZE = 5
_MIN_REGION_SIZE = 8
_DEFAULT_REF_SIZE = 96
_DOUBLE_CLICK_INTERVAL_MS = 350


class ROIConfiguratorError(RuntimeError):
    pass


@dataclass(slots=True)
class RectRegion:
    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height

    def normalized(self) -> "RectRegion":
        x0 = min(self.x, self.x1)
        y0 = min(self.y, self.y1)
        x1 = max(self.x, self.x1)
        y1 = max(self.y, self.y1)
        return RectRegion(x=x0, y=y0, width=max(0, x1 - x0), height=max(0, y1 - y0))

    def clip(self, image_width: int, image_height: int, min_size: int = _MIN_REGION_SIZE) -> "RectRegion":
        rect = self.normalized()
        x0 = int(np.clip(rect.x, 0, max(image_width - 1, 0)))
        y0 = int(np.clip(rect.y, 0, max(image_height - 1, 0)))
        x1 = int(np.clip(rect.x1, x0 + 1, image_width))
        y1 = int(np.clip(rect.y1, y0 + 1, image_height))
        if x1 - x0 < min_size:
            x1 = min(image_width, x0 + min_size)
            x0 = max(0, x1 - min_size)
        if y1 - y0 < min_size:
            y1 = min(image_height, y0 + min_size)
            y0 = max(0, y1 - min_size)
        return RectRegion(x=x0, y=y0, width=max(1, x1 - x0), height=max(1, y1 - y0))

    def contains(self, px: int, py: int) -> bool:
        rect = self.normalized()
        return rect.x <= px <= rect.x1 and rect.y <= py <= rect.y1


@dataclass(slots=True)
class ReferenceRegionState:
    name: str
    rect: RectRegion
    weight: float = 1.0


@dataclass(slots=True)
class ROISelectionState:
    frame_bgr: np.ndarray
    frame_gray: np.ndarray
    display_scale: float
    image_width: int
    image_height: int
    roi: RectRegion | None
    references: list[ReferenceRegionState]
    active_type: str
    active_index: int
    drag_mode: str | None
    drag_anchor: tuple[int, int] | None
    drag_start_rect: RectRegion | None
    is_dirty: bool
    last_click_time_ms: int
    pending_single_click: bool


@dataclass(slots=True)
class ROIConfiguratorResult:
    frame_source: str
    frame_index: int
    roi: tuple[int, int, int, int]
    reference_regions: list[ReferenceRegion]
    config_path: Path


@dataclass(slots=True)
class FramePreview:
    frame_bgr: np.ndarray
    frame_gray: np.ndarray
    frame_source: str
    frame_index: int


_HANDLE_KEYS = {
    "tl",
    "tr",
    "bl",
    "br",
    "l",
    "r",
    "t",
    "b",
}


def _resize_for_preview(frame: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= _MAX_PREVIEW_WIDTH:
        return frame.copy(), 1.0
    scale = _MAX_PREVIEW_WIDTH / float(width)
    resized = cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _load_frame_preview(config: ProjectConfig, frame_index: int | None = None) -> FramePreview:
    raw_video = Path(config.paths.raw_video)
    if raw_video.exists():
        sequence = load_video_gray(raw_video)
        index = 0 if frame_index is None else int(np.clip(frame_index, 0, sequence.frames.shape[0] - 1))
        frame_gray = np.clip(sequence.frames[index], 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        return FramePreview(
            frame_bgr=frame_bgr,
            frame_gray=frame_gray,
            frame_source=f"video:{raw_video}",
            frame_index=index,
        )

    camera_cfg = config.camera
    backend = camera_cfg.backend if camera_cfg.backend is not None else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_cfg.camera_index, backend)
    if not cap.isOpened():
        raise ROIConfiguratorError(
            f"无法打开视频 {raw_video}，且无法打开相机索引 {camera_cfg.camera_index} 读取预览帧"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
    cap.set(cv2.CAP_PROP_FPS, camera_cfg.fps)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ROIConfiguratorError("无法从相机读取预览帧")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    return FramePreview(
        frame_bgr=frame_bgr,
        frame_gray=frame_gray,
        frame_source=f"camera:{camera_cfg.camera_index}",
        frame_index=0,
    )


def _clip_rect(rect: RectRegion, width: int, height: int) -> RectRegion:
    return rect.clip(width, height, min_size=_MIN_REGION_SIZE)


def _rect_from_tuple(values: tuple[int, int, int, int] | None) -> RectRegion | None:
    if values is None:
        return None
    x, y, w, h = values
    return RectRegion(int(x), int(y), int(w), int(h)).normalized()


def _reference_states_from_config(config: ProjectConfig) -> list[ReferenceRegionState]:
    refs: list[ReferenceRegionState] = []
    for region in config.reference_regions:
        refs.append(
            ReferenceRegionState(
                name=region.name,
                rect=RectRegion(region.x, region.y, region.width, region.height).normalized(),
                weight=region.weight,
            )
        )
    return refs


def _make_default_roi(image_width: int, image_height: int) -> RectRegion:
    width = max(_MIN_REGION_SIZE, int(round(image_width * 0.45)))
    height = max(_MIN_REGION_SIZE, int(round(image_height * 0.45)))
    x = max(0, (image_width - width) // 2)
    y = max(0, (image_height - height) // 2)
    return RectRegion(x, y, width, height)


def _make_default_reference(image_width: int, image_height: int, count: int) -> RectRegion:
    size = min(_DEFAULT_REF_SIZE, max(_MIN_REGION_SIZE, min(image_width, image_height) // 5))
    margin = max(10, size // 3)
    columns = max(1, (image_width - margin) // max(size + margin, 1))
    col = count % columns
    row = count // columns
    x = margin + col * (size + margin)
    y = margin + row * (size + margin)
    x = min(max(0, x), max(0, image_width - size))
    y = min(max(0, y), max(0, image_height - size))
    return RectRegion(x, y, size, size)


def _generate_reference_name(existing: list[ReferenceRegionState]) -> str:
    used = {item.name for item in existing}
    index = 1
    while True:
        name = f"ref_{index}"
        if name not in used:
            return name
        index += 1


def _image_to_screen(point: tuple[int, int], scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def _screen_to_image(point: tuple[int, int], scale: float, image_width: int, image_height: int) -> tuple[int, int]:
    inv = 1.0 / max(scale, 1e-6)
    x = int(round(point[0] * inv))
    y = int(round(point[1] * inv))
    x = int(np.clip(x, 0, max(0, image_width - 1)))
    y = int(np.clip(y, 0, max(0, image_height - 1)))
    return x, y


def _panel_x_offset(state: ROISelectionState) -> int:
    return int(round(state.image_width * state.display_scale))


def _rect_screen_bounds(rect: RectRegion, scale: float) -> tuple[int, int, int, int]:
    x0, y0 = _image_to_screen((rect.x, rect.y), scale)
    x1, y1 = _image_to_screen((rect.x1, rect.y1), scale)
    return x0, y0, x1, y1


def _handle_positions(rect: RectRegion) -> dict[str, tuple[int, int]]:
    return {
        "tl": (rect.x, rect.y),
        "tr": (rect.x1, rect.y),
        "bl": (rect.x, rect.y1),
        "br": (rect.x1, rect.y1),
        "l": (rect.x, rect.y + rect.height // 2),
        "r": (rect.x1, rect.y + rect.height // 2),
        "t": (rect.x + rect.width // 2, rect.y),
        "b": (rect.x + rect.width // 2, rect.y1),
    }


def _hit_test_rect(rect: RectRegion, px: int, py: int, scale: float) -> str | None:
    handle_radius = max(8, int(round(10 / max(scale, 1e-6))))
    for key, (hx, hy) in _handle_positions(rect).items():
        if abs(px - hx) <= handle_radius and abs(py - hy) <= handle_radius:
            return key
    if rect.contains(px, py):
        return "move"
    return None


def _hit_test(state: ROISelectionState, px: int, py: int) -> tuple[str | None, int, str | None]:
    if state.roi is not None:
        roi_mode = _hit_test_rect(state.roi, px, py, state.display_scale)
        if roi_mode is not None:
            return "roi", -1, roi_mode
    for idx in range(len(state.references) - 1, -1, -1):
        mode = _hit_test_rect(state.references[idx].rect, px, py, state.display_scale)
        if mode is not None:
            return "ref", idx, mode
    return None, -1, None


def _set_active(state: ROISelectionState, active_type: str | None, active_index: int = -1) -> None:
    state.active_type = active_type or "none"
    state.active_index = active_index


def _draw_handles(canvas: np.ndarray, rect: RectRegion, scale: float, color: tuple[int, int, int]) -> None:
    for point in _handle_positions(rect).values():
        sx, sy = _image_to_screen(point, scale)
        cv2.rectangle(
            canvas,
            (sx - _HANDLE_HALF_SIZE, sy - _HANDLE_HALF_SIZE),
            (sx + _HANDLE_HALF_SIZE, sy + _HANDLE_HALF_SIZE),
            color,
            -1,
        )
        cv2.rectangle(
            canvas,
            (sx - _HANDLE_HALF_SIZE, sy - _HANDLE_HALF_SIZE),
            (sx + _HANDLE_HALF_SIZE, sy + _HANDLE_HALF_SIZE),
            (0, 0, 0),
            1,
        )


def _draw_region(
    canvas: np.ndarray,
    rect: RectRegion,
    scale: float,
    color: tuple[int, int, int],
    label: str,
    selected: bool,
) -> None:
    x0, y0, x1, y1 = _rect_screen_bounds(rect, scale)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    alpha = 0.18 if selected else 0.12
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3 if selected else 2)
    cv2.putText(canvas, label, (x0 + 6, max(20, y0 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (x0 + 6, max(20, y0 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if selected:
        _draw_handles(canvas, rect, scale, _HOVER_COLOR)


def _draw_status_panel(canvas: np.ndarray, state: ROISelectionState, frame_source: str, frame_index: int) -> None:
    image_screen_width = int(round(state.image_width * state.display_scale))
    panel_left = image_screen_width
    cv2.rectangle(canvas, (panel_left, 0), (canvas.shape[1], canvas.shape[0]), _PANEL_BG, -1)
    cv2.line(canvas, (panel_left, 0), (panel_left, canvas.shape[0]), (70, 70, 70), 1)

    lines = [
        "Interactive ROI & Reference Configurator",
        f"Frame source: {frame_source}",
        f"Frame index: {frame_index}",
        "",
    ]
    if state.roi is not None:
        lines.append(f"ROI: x={state.roi.x}, y={state.roi.y}, w={state.roi.width}, h={state.roi.height}")
    else:
        lines.append("ROI: not set")
    lines.append(f"Reference regions: {len(state.references)}")
    if state.active_type == "ref" and 0 <= state.active_index < len(state.references):
        ref = state.references[state.active_index]
        lines.append(
            f"Selected ref: {ref.name} x={ref.rect.x}, y={ref.rect.y}, w={ref.rect.width}, h={ref.rect.height}, weight={ref.weight:.2f}"
        )
    elif state.active_type == "roi" and state.roi is not None:
        lines.append("Selected: ROI")
    else:
        lines.append("Selected: none")
    lines.extend(
        [
            "",
            "Mouse:",
            "- Drag ROI/ref rectangle to move",
            "- Drag white handles to resize",
            "- Double click image blank area to create ROI if missing",
            "",
            "Keyboard:",
            "- A: add reference region",
            "- TAB: cycle selection",
            "- R: select ROI",
            "- D/Delete: delete selected reference",
            "- C: clear all references",
            "- I/K/J/L: nudge selected region",
            "- Shift + I/K/J/L: resize selected region",
            "- S/Enter: save to YAML",
            "- Esc/Q: quit without saving",
        ]
    )
    y = 32
    for line in lines:
        color = _PANEL_ACCENT if line.startswith("Interactive") else _TEXT_COLOR
        cv2.putText(canvas, line, (panel_left + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (panel_left + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.57, color, 1, cv2.LINE_AA)
        y += 24

    dirty_text = "Modified" if state.is_dirty else "Synced"
    dirty_color = (0, 210, 255) if state.is_dirty else (0, 220, 0)
    footer_y = canvas.shape[0] - 24
    cv2.putText(canvas, dirty_text, (panel_left + 16, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, dirty_text, (panel_left + 16, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dirty_color, 2, cv2.LINE_AA)


def _render_configurator(state: ROISelectionState, frame_source: str, frame_index: int) -> np.ndarray:
    preview, _ = _resize_for_preview(state.frame_bgr)
    canvas = np.zeros((preview.shape[0], preview.shape[1] + _PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, : preview.shape[1]] = preview
    if state.roi is not None:
        _draw_region(canvas, state.roi, state.display_scale, _ROI_COLOR, "ROI", state.active_type == "roi")
    for idx, ref in enumerate(state.references):
        selected = state.active_type == "ref" and idx == state.active_index
        _draw_region(canvas, ref.rect, state.display_scale, _REF_COLOR if not selected else _ACTIVE_COLOR, ref.name, selected)
    _draw_status_panel(canvas, state, frame_source, frame_index)
    return canvas


def _apply_drag(rect: RectRegion, drag_mode: str, start_rect: RectRegion, start_point: tuple[int, int], point: tuple[int, int]) -> RectRegion:
    dx = point[0] - start_point[0]
    dy = point[1] - start_point[1]
    x0 = start_rect.x
    y0 = start_rect.y
    x1 = start_rect.x1
    y1 = start_rect.y1

    if drag_mode == "move":
        return RectRegion(x0 + dx, y0 + dy, start_rect.width, start_rect.height)
    if drag_mode == "tl":
        return RectRegion(x0 + dx, y0 + dy, x1 - (x0 + dx), y1 - (y0 + dy))
    if drag_mode == "tr":
        return RectRegion(x0, y0 + dy, (x1 + dx) - x0, y1 - (y0 + dy))
    if drag_mode == "bl":
        return RectRegion(x0 + dx, y0, x1 - (x0 + dx), (y1 + dy) - y0)
    if drag_mode == "br":
        return RectRegion(x0, y0, (x1 + dx) - x0, (y1 + dy) - y0)
    if drag_mode == "l":
        return RectRegion(x0 + dx, y0, x1 - (x0 + dx), start_rect.height)
    if drag_mode == "r":
        return RectRegion(x0, y0, (x1 + dx) - x0, start_rect.height)
    if drag_mode == "t":
        return RectRegion(x0, y0 + dy, start_rect.width, y1 - (y0 + dy))
    if drag_mode == "b":
        return RectRegion(x0, y0, start_rect.width, (y1 + dy) - y0)
    return start_rect


def _selected_rect(state: ROISelectionState) -> RectRegion | None:
    if state.active_type == "roi":
        return state.roi
    if state.active_type == "ref" and 0 <= state.active_index < len(state.references):
        return state.references[state.active_index].rect
    return None


def _set_selected_rect(state: ROISelectionState, rect: RectRegion) -> None:
    clipped = _clip_rect(rect, state.image_width, state.image_height)
    if state.active_type == "roi":
        state.roi = clipped
        state.is_dirty = True
        return
    if state.active_type == "ref" and 0 <= state.active_index < len(state.references):
        state.references[state.active_index].rect = clipped
        state.is_dirty = True


def _cycle_selection(state: ROISelectionState) -> None:
    order: list[tuple[str, int]] = []
    if state.roi is not None:
        order.append(("roi", -1))
    order.extend(("ref", idx) for idx in range(len(state.references)))
    if not order:
        _set_active(state, None, -1)
        return
    current = (state.active_type, state.active_index)
    if current not in order:
        _set_active(state, order[0][0], order[0][1])
        return
    idx = (order.index(current) + 1) % len(order)
    _set_active(state, order[idx][0], order[idx][1])


def _add_reference_region(state: ROISelectionState) -> None:
    name = _generate_reference_name(state.references)
    rect = _make_default_reference(state.image_width, state.image_height, len(state.references))
    state.references.append(ReferenceRegionState(name=name, rect=rect, weight=1.0))
    _set_active(state, "ref", len(state.references) - 1)
    state.is_dirty = True


def _delete_selected_reference(state: ROISelectionState) -> None:
    if state.active_type != "ref" or not (0 <= state.active_index < len(state.references)):
        return
    del state.references[state.active_index]
    if state.references:
        next_index = min(state.active_index, len(state.references) - 1)
        _set_active(state, "ref", next_index)
    elif state.roi is not None:
        _set_active(state, "roi", -1)
    else:
        _set_active(state, None, -1)
    state.is_dirty = True


def _clear_references(state: ROISelectionState) -> None:
    if not state.references:
        return
    state.references.clear()
    if state.roi is not None:
        _set_active(state, "roi", -1)
    else:
        _set_active(state, None, -1)
    state.is_dirty = True


def _nudge_selected(state: ROISelectionState, dx: int, dy: int, resize: bool) -> None:
    rect = _selected_rect(state)
    if rect is None:
        return
    if resize:
        updated = RectRegion(rect.x, rect.y, rect.width + dx, rect.height + dy)
    else:
        updated = RectRegion(rect.x + dx, rect.y + dy, rect.width, rect.height)
    _set_selected_rect(state, updated)


def _write_config_regions(config_path: Path, state: ROISelectionState) -> None:
    if state.roi is None:
        raise ROIConfiguratorError("保存前必须先定义 ROI")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload.setdefault("dic", {})
    payload["dic"]["roi"] = [int(state.roi.x), int(state.roi.y), int(state.roi.width), int(state.roi.height)]
    payload["reference_regions"] = [
        {
            "name": ref.name,
            "x": int(ref.rect.x),
            "y": int(ref.rect.y),
            "width": int(ref.rect.width),
            "height": int(ref.rect.height),
            "weight": float(ref.weight),
        }
        for ref in state.references
    ]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def configure_roi_interactively(
    config_path: str | Path,
    frame_index: int | None = None,
) -> ROIConfiguratorResult:
    path = Path(config_path)
    config = load_config(path)
    preview = _load_frame_preview(config, frame_index=frame_index)
    preview_resized, scale = _resize_for_preview(preview.frame_bgr)
    image_height, image_width = preview.frame_gray.shape[:2]

    state = ROISelectionState(
        frame_bgr=preview.frame_bgr,
        frame_gray=preview.frame_gray,
        display_scale=scale,
        image_width=image_width,
        image_height=image_height,
        roi=_rect_from_tuple(config.dic.roi) or _make_default_roi(image_width, image_height),
        references=_reference_states_from_config(config),
        active_type="roi",
        active_index=-1,
        drag_mode=None,
        drag_anchor=None,
        drag_start_rect=None,
        is_dirty=False,
        last_click_time_ms=0,
        pending_single_click=False,
    )
    if state.references and config.dic.roi is None:
        _set_active(state, "ref", 0)

    def on_mouse(event: int, x: int, y: int, flags: int, *_: Any) -> None:
        nonlocal state
        image_screen_width = preview_resized.shape[1]
        if x >= image_screen_width or y >= preview_resized.shape[0]:
            return
        image_point = _screen_to_image((x, y), state.display_scale, state.image_width, state.image_height)

        if event == cv2.EVENT_LBUTTONDOWN:
            active_type, active_index, drag_mode = _hit_test(state, image_point[0], image_point[1])
            now_ms = cv2.getTickCount() * 1000 // cv2.getTickFrequency()
            is_double_click = (now_ms - state.last_click_time_ms) <= _DOUBLE_CLICK_INTERVAL_MS
            state.last_click_time_ms = int(now_ms)
            if active_type is None:
                if is_double_click:
                    state.roi = _clip_rect(
                        RectRegion(image_point[0] - 80, image_point[1] - 80, 160, 160),
                        state.image_width,
                        state.image_height,
                    )
                    _set_active(state, "roi", -1)
                    state.is_dirty = True
                else:
                    _set_active(state, None, -1)
                return
            _set_active(state, active_type, active_index)
            state.drag_mode = drag_mode
            state.drag_anchor = image_point
            state.drag_start_rect = _selected_rect(state)
            return

        if event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if state.drag_mode is None or state.drag_anchor is None or state.drag_start_rect is None:
                return
            updated = _apply_drag(state.drag_start_rect, state.drag_mode, state.drag_start_rect, state.drag_anchor, image_point)
            _set_selected_rect(state, updated)
            return

        if event == cv2.EVENT_LBUTTONUP:
            state.drag_mode = None
            state.drag_anchor = None
            state.drag_start_rect = None

    cv2.namedWindow(_CONFIG_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(_CONFIG_WINDOW_NAME, min(preview_resized.shape[1] + _PANEL_WIDTH, 1800), min(preview_resized.shape[0] + 80, 1100))
    cv2.setMouseCallback(_CONFIG_WINDOW_NAME, on_mouse)

    saved = False
    try:
        while True:
            canvas = _render_configurator(state, preview.frame_source, preview.frame_index)
            cv2.imshow(_CONFIG_WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (13, ord("s")):
                _write_config_regions(path, state)
                state.is_dirty = False
                saved = True
                break
            if key == ord("a"):
                _add_reference_region(state)
                continue
            if key == 9:
                _cycle_selection(state)
                continue
            if key == ord("r"):
                if state.roi is None:
                    state.roi = _make_default_roi(state.image_width, state.image_height)
                    state.is_dirty = True
                _set_active(state, "roi", -1)
                continue
            if key in (ord("d"), 127):
                _delete_selected_reference(state)
                continue
            if key == ord("c"):
                _clear_references(state)
                continue
            if key == ord("j"):
                _nudge_selected(state, dx=-1, dy=0, resize=False)
                continue
            if key == ord("l"):
                _nudge_selected(state, dx=1, dy=0, resize=False)
                continue
            if key == ord("i"):
                _nudge_selected(state, dx=0, dy=-1, resize=False)
                continue
            if key == ord("k"):
                _nudge_selected(state, dx=0, dy=1, resize=False)
                continue
            if key == ord("J"):
                _nudge_selected(state, dx=-1, dy=0, resize=True)
                continue
            if key == ord("L"):
                _nudge_selected(state, dx=1, dy=0, resize=True)
                continue
            if key == ord("I"):
                _nudge_selected(state, dx=0, dy=-1, resize=True)
                continue
            if key == ord("K"):
                _nudge_selected(state, dx=0, dy=1, resize=True)
                continue
        if not saved:
            raise ROIConfiguratorError("用户取消了 ROI/reference_regions 配置，未写入 YAML")
    finally:
        cv2.destroyAllWindows()

    if state.roi is None:
        raise ROIConfiguratorError("未定义 ROI，无法生成配置结果")

    references = [
        ReferenceRegion(name=ref.name, x=ref.rect.x, y=ref.rect.y, width=ref.rect.width, height=ref.rect.height, weight=ref.weight)
        for ref in state.references
    ]
    return ROIConfiguratorResult(
        frame_source=preview.frame_source,
        frame_index=preview.frame_index,
        roi=(state.roi.x, state.roi.y, state.roi.width, state.roi.height),
        reference_regions=references,
        config_path=path,
    )
