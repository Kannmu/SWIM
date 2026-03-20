from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


ColorRGB = tuple[int, int, int]
Point = tuple[int, int]
Rect = tuple[int, int, int, int]


@dataclass(slots=True)
class ThemeColors:
    background: str = "#0f172a"
    surface: str = "#111827"
    surface_alt: str = "#1f2937"
    panel: str = "#0b1220"
    border: str = "#334155"
    text: str = "#e5e7eb"
    muted_text: str = "#94a3b8"
    accent: str = "#3b82f6"
    accent_hover: str = "#2563eb"
    success: str = "#22c55e"
    warning: str = "#f59e0b"
    danger: str = "#ef4444"
    roi: str = "#22c55e"
    reference: str = "#f59e0b"
    active: str = "#60a5fa"


@dataclass(slots=True)
class DragState:
    mode: str | None = None
    target: str | None = None
    index: int = -1
    start_point: Point | None = None
    start_rect: Rect | None = None


def hex_to_rgb(value: str) -> ColorRGB:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"invalid color string: {value}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def resize_to_fit(frame: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame.copy(), 1.0
    scale = min(max_width / width, max_height / height)
    scale = min(scale, 1.0)
    if scale >= 0.999:
        return frame.copy(), 1.0
    resized = cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def screen_to_image(point: Point, scale: float, image_width: int, image_height: int) -> Point:
    inv = 1.0 / max(scale, 1e-6)
    x = clamp(int(round(point[0] * inv)), 0, max(image_width - 1, 0))
    y = clamp(int(round(point[1] * inv)), 0, max(image_height - 1, 0))
    return x, y


def image_to_screen(point: Point, scale: float) -> Point:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def rect_to_screen(rect: Rect, scale: float) -> Rect:
    x, y, width, height = rect
    x0, y0 = image_to_screen((x, y), scale)
    x1, y1 = image_to_screen((x + width, y + height), scale)
    return x0, y0, x1 - x0, y1 - y0


def fit_rect_to_bounds(rect: Rect, image_width: int, image_height: int, min_size: int = 8) -> Rect:
    x, y, width, height = rect
    x0 = min(x, x + width)
    y0 = min(y, y + height)
    x1 = max(x, x + width)
    y1 = max(y, y + height)

    x0 = clamp(x0, 0, max(image_width - 1, 0))
    y0 = clamp(y0, 0, max(image_height - 1, 0))
    x1 = clamp(x1, x0 + 1, image_width)
    y1 = clamp(y1, y0 + 1, image_height)

    if x1 - x0 < min_size:
        x1 = min(image_width, x0 + min_size)
        x0 = max(0, x1 - min_size)
    if y1 - y0 < min_size:
        y1 = min(image_height, y0 + min_size)
        y0 = max(0, y1 - min_size)
    return x0, y0, x1 - x0, y1 - y0


def move_or_resize_rect(start_rect: Rect, drag_mode: str, start_point: Point, point: Point) -> Rect:
    dx = point[0] - start_point[0]
    dy = point[1] - start_point[1]
    x, y, width, height = start_rect
    x1 = x + width
    y1 = y + height

    if drag_mode == "move":
        return x + dx, y + dy, width, height
    if drag_mode == "tl":
        return x + dx, y + dy, x1 - (x + dx), y1 - (y + dy)
    if drag_mode == "tr":
        return x, y + dy, (x1 + dx) - x, y1 - (y + dy)
    if drag_mode == "bl":
        return x + dx, y, x1 - (x + dx), (y1 + dy) - y
    if drag_mode == "br":
        return x, y, (x1 + dx) - x, (y1 + dy) - y
    if drag_mode == "l":
        return x + dx, y, x1 - (x + dx), height
    if drag_mode == "r":
        return x, y, (x1 + dx) - x, height
    if drag_mode == "t":
        return x, y + dy, width, y1 - (y + dy)
    if drag_mode == "b":
        return x, y, width, (y1 + dy) - y
    return start_rect


def handle_positions(rect: Rect) -> dict[str, Point]:
    x, y, width, height = rect
    x1 = x + width
    y1 = y + height
    return {
        "tl": (x, y),
        "tr": (x1, y),
        "bl": (x, y1),
        "br": (x1, y1),
        "l": (x, y + height // 2),
        "r": (x1, y + height // 2),
        "t": (x + width // 2, y),
        "b": (x + width // 2, y1),
    }


def hit_test_rect(rect: Rect, point: Point, scale: float) -> str | None:
    px, py = point
    radius = max(8, int(round(10 / max(scale, 1e-6))))
    for key, (hx, hy) in handle_positions(rect).items():
        if abs(px - hx) <= radius and abs(py - hy) <= radius:
            return key
    x, y, width, height = rect
    if x <= px <= x + width and y <= py <= y + height:
        return "move"
    return None


def draw_labeled_rect(
    image: np.ndarray,
    rect: Rect,
    scale: float,
    color: ColorRGB,
    label: str,
    selected: bool,
    fill_alpha: float = 0.16,
) -> None:
    x, y, width, height = rect_to_screen(rect, scale)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
    cv2.addWeighted(overlay, fill_alpha, image, 1.0 - fill_alpha, 0.0, image)
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 3 if selected else 2)
    cv2.putText(image, label, (x + 8, max(24, y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (x + 8, max(24, y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if not selected:
        return
    for px, py in handle_positions(rect).values():
        sx, sy = image_to_screen((px, py), scale)
        cv2.rectangle(image, (sx - 5, sy - 5), (sx + 5, sy + 5), (255, 255, 255), -1)
        cv2.rectangle(image, (sx - 5, sy - 5), (sx + 5, sy + 5), (15, 23, 42), 1)


def frame_to_ctk_image(frame_bgr: np.ndarray, size: tuple[int, int] | None = None) -> ctk.CTkImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    target_size = size if size is not None else image.size
    return ctk.CTkImage(light_image=image, dark_image=image, size=target_size)


def set_widget_text(widget: ctk.CTkTextbox, text: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.configure(state="disabled")


class CTkImageCanvas(ctk.CTkFrame):
    def __init__(
        self,
        master,
        width: int,
        height: int,
        click_callback: Callable[[int, int], None] | None = None,
        drag_callback: Callable[[str, int, int], None] | None = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._width = width
        self._height = height
        self._photo: ImageTk.PhotoImage | None = None
        self._click_callback = click_callback
        self._drag_callback = drag_callback
        self.canvas = ctk.CTkCanvas(
            self,
            width=width,
            height=height,
            highlightthickness=0,
            bg="#000000",
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

    def configure_callbacks(
        self,
        click_callback: Callable[[int, int], None] | None = None,
        drag_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self._click_callback = click_callback
        self._drag_callback = drag_callback

    def set_image(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.canvas.configure(width=image.width, height=image.height)
        self._width = image.width
        self._height = image.height

    def _on_press(self, event) -> None:
        if self._drag_callback is not None:
            self._drag_callback("press", int(event.x), int(event.y))
        elif self._click_callback is not None:
            self._click_callback(int(event.x), int(event.y))

    def _on_drag(self, event) -> None:
        if self._drag_callback is not None:
            self._drag_callback("move", int(event.x), int(event.y))

    def _on_release(self, event) -> None:
        if self._drag_callback is not None:
            self._drag_callback("release", int(event.x), int(event.y))

    def _on_double_click(self, event) -> None:
        if self._drag_callback is not None:
            self._drag_callback("double", int(event.x), int(event.y))
