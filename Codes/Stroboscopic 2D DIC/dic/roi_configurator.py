from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
import yaml
from tkinter import messagebox

from .config import ProjectConfig, ReferenceRegion, load_config
from .gui_common import (
    CTkImageCanvas,
    ThemeColors,
    draw_labeled_rect,
    fit_rect_to_bounds,
    hit_test_rect,
    image_to_screen,
    move_or_resize_rect,
    resize_to_fit,
    screen_to_image,
    set_widget_text,
)
from .io_utils import load_video_gray


_MIN_REGION_SIZE = 8
_DEFAULT_REF_SIZE = 96
_PREVIEW_MAX_WIDTH = 1280
_PREVIEW_MAX_HEIGHT = 860
_THEME = ThemeColors()


class ROIConfiguratorError(RuntimeError):
    pass


@dataclass(slots=True)
class ReferenceRegionState:
    name: str
    rect: tuple[int, int, int, int]
    weight: float = 1.0


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


def _rect_from_tuple(values: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if values is None:
        return None
    return fit_rect_to_bounds(tuple(int(v) for v in values), 100000, 100000, min_size=_MIN_REGION_SIZE)


def _reference_states_from_config(config: ProjectConfig) -> list[ReferenceRegionState]:
    refs: list[ReferenceRegionState] = []
    for region in config.reference_regions:
        refs.append(
            ReferenceRegionState(
                name=region.name,
                rect=(region.x, region.y, region.width, region.height),
                weight=region.weight,
            )
        )
    return refs


def _make_default_roi(image_width: int, image_height: int) -> tuple[int, int, int, int]:
    width = max(_MIN_REGION_SIZE, int(round(image_width * 0.45)))
    height = max(_MIN_REGION_SIZE, int(round(image_height * 0.45)))
    x = max(0, (image_width - width) // 2)
    y = max(0, (image_height - height) // 2)
    return x, y, width, height


def _make_default_reference(image_width: int, image_height: int, count: int) -> tuple[int, int, int, int]:
    size = min(_DEFAULT_REF_SIZE, max(_MIN_REGION_SIZE, min(image_width, image_height) // 5))
    margin = max(10, size // 3)
    columns = max(1, (image_width - margin) // max(size + margin, 1))
    col = count % columns
    row = count // columns
    x = margin + col * (size + margin)
    y = margin + row * (size + margin)
    return fit_rect_to_bounds((x, y, size, size), image_width, image_height, min_size=_MIN_REGION_SIZE)


def _generate_reference_name(existing: list[ReferenceRegionState]) -> str:
    used = {item.name for item in existing}
    index = 1
    while True:
        name = f"ref_{index}"
        if name not in used:
            return name
        index += 1


def _write_config_regions(
    config_path: Path,
    roi: tuple[int, int, int, int] | None,
    references: list[ReferenceRegionState],
) -> None:
    if roi is None:
        raise ROIConfiguratorError("保存前必须先定义 ROI")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload.setdefault("dic", {})
    payload["dic"]["roi"] = [int(v) for v in roi]
    payload["reference_regions"] = [
        {
            "name": ref.name,
            "x": int(ref.rect[0]),
            "y": int(ref.rect[1]),
            "width": int(ref.rect[2]),
            "height": int(ref.rect[3]),
            "weight": float(ref.weight),
        }
        for ref in references
    ]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


class ROIConfiguratorApp(ctk.CTk):
    def __init__(self, config_path: Path, preview: FramePreview, config: ProjectConfig):
        super().__init__()
        self.config_path = config_path
        self.preview = preview
        self.image_height, self.image_width = preview.frame_bgr.shape[:2]
        self.display_frame, self.display_scale = resize_to_fit(preview.frame_bgr, _PREVIEW_MAX_WIDTH, _PREVIEW_MAX_HEIGHT)
        self.preview_height, self.preview_width = self.display_frame.shape[:2]

        self.roi = _rect_from_tuple(config.dic.roi) or _make_default_roi(self.image_width, self.image_height)
        self.references = _reference_states_from_config(config)
        self.active_type = "roi"
        self.active_index = -1
        self.drag_mode: str | None = None
        self.drag_start_point: tuple[int, int] | None = None
        self.drag_start_rect: tuple[int, int, int, int] | None = None
        self.is_dirty = False
        self.saved = False
        self.result: ROIConfiguratorResult | None = None

        self.title("SWIM ROI Configurator")
        self.geometry(f"{self.preview_width + 430}x{max(self.preview_height + 80, 860)}")
        self.minsize(1180, 820)
        self.configure(fg_color=_THEME.background)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.bind("<Escape>", lambda _event: self.cancel())
        self.bind("<KeyPress-q>", lambda _event: self.cancel())
        self.bind("<KeyPress-Q>", lambda _event: self.cancel())
        self.bind("<KeyPress-s>", lambda _event: self.save())
        self.bind("<Return>", lambda _event: self.save())
        self.bind("<Tab>", self._on_tab_key)
        self.bind("<KeyPress-a>", lambda _event: self.add_reference())
        self.bind("<KeyPress-A>", lambda _event: self.add_reference())
        self.bind("<KeyPress-r>", lambda _event: self.select_roi())
        self.bind("<KeyPress-R>", lambda _event: self.select_roi())
        self.bind("<KeyPress-c>", lambda _event: self.clear_references())
        self.bind("<KeyPress-C>", lambda _event: self.clear_references())
        self.bind("<Delete>", lambda _event: self.delete_selected_reference())
        self.bind("<BackSpace>", lambda _event: self.delete_selected_reference())
        self.bind("<KeyPress-d>", lambda _event: self.delete_selected_reference())
        self.bind("<KeyPress-D>", lambda _event: self.delete_selected_reference())
        self.bind("<KeyPress-i>", lambda _event: self.nudge_selected(0, -1, False))
        self.bind("<KeyPress-k>", lambda _event: self.nudge_selected(0, 1, False))
        self.bind("<KeyPress-j>", lambda _event: self.nudge_selected(-1, 0, False))
        self.bind("<KeyPress-l>", lambda _event: self.nudge_selected(1, 0, False))
        self.bind("<KeyPress-I>", lambda _event: self.nudge_selected(0, -1, True))
        self.bind("<KeyPress-K>", lambda _event: self.nudge_selected(0, 1, True))
        self.bind("<KeyPress-J>", lambda _event: self.nudge_selected(-1, 0, True))
        self.bind("<KeyPress-L>", lambda _event: self.nudge_selected(1, 0, True))

        self._build_layout()
        self.refresh_view()

    def _build_layout(self) -> None:
        viewer_card = ctk.CTkFrame(self, corner_radius=18, fg_color=_THEME.surface, border_width=1, border_color=_THEME.border)
        viewer_card.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
        viewer_card.grid_rowconfigure(1, weight=1)
        viewer_card.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(viewer_card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="ROI 与参考区域配置器",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=_THEME.text,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="拖拽矩形或控制点完成交互式配置，并直接写回 YAML",
            font=ctk.CTkFont(size=13),
            text_color=_THEME.muted_text,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.canvas = CTkImageCanvas(viewer_card, width=self.preview_width, height=self.preview_height, drag_callback=self._handle_canvas_event)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))

        side = ctk.CTkFrame(self, width=390, corner_radius=18, fg_color=_THEME.panel, border_width=1, border_color=_THEME.border)
        side.grid(row=0, column=1, sticky="ns", padx=(10, 18), pady=18)
        side.grid_propagate(False)

        ctk.CTkLabel(side, text="控制面板", font=ctk.CTkFont(size=20, weight="bold"), text_color=_THEME.text).pack(anchor="w", padx=18, pady=(18, 6))
        ctk.CTkLabel(side, text="适用于拍摄预览图、ROI 选区与 reference_regions 配置", font=ctk.CTkFont(size=12), text_color=_THEME.muted_text, justify="left").pack(anchor="w", padx=18)

        meta_card = ctk.CTkFrame(side, corner_radius=14, fg_color=_THEME.surface_alt)
        meta_card.pack(fill="x", padx=18, pady=(16, 12))
        self.meta_label = ctk.CTkLabel(meta_card, text="", justify="left", anchor="w", font=ctk.CTkFont(size=13), text_color=_THEME.text)
        self.meta_label.pack(fill="x", padx=14, pady=14)

        action_row = ctk.CTkFrame(side, fg_color="transparent")
        action_row.pack(fill="x", padx=18, pady=(0, 12))
        action_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(action_row, text="新增参考区 (A)", command=self.add_reference, fg_color=_THEME.accent, hover_color=_THEME.accent_hover).grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(action_row, text="选择 ROI (R)", command=self.select_roi, fg_color="#16a34a", hover_color="#15803d").grid(row=0, column=1, padx=(6, 0), sticky="ew")

        action_row2 = ctk.CTkFrame(side, fg_color="transparent")
        action_row2.pack(fill="x", padx=18, pady=(0, 12))
        action_row2.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(action_row2, text="删除选中参考区", command=self.delete_selected_reference, fg_color="#dc2626", hover_color="#b91c1c").grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(action_row2, text="清空参考区 (C)", command=self.clear_references, fg_color="#f59e0b", hover_color="#d97706").grid(row=0, column=1, padx=(6, 0), sticky="ew")

        refs_card = ctk.CTkFrame(side, corner_radius=14, fg_color=_THEME.surface_alt)
        refs_card.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        ctk.CTkLabel(refs_card, text="参考区列表", font=ctk.CTkFont(size=16, weight="bold"), text_color=_THEME.text).pack(anchor="w", padx=14, pady=(12, 6))
        self.refs_text = ctk.CTkTextbox(refs_card, height=220, font=ctk.CTkFont(family="Consolas", size=12), activate_scrollbars=True)
        self.refs_text.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.refs_text.configure(state="disabled")

        help_card = ctk.CTkFrame(side, corner_radius=14, fg_color=_THEME.surface_alt)
        help_card.pack(fill="x", padx=18, pady=(0, 12))
        ctk.CTkLabel(help_card, text="操作说明", font=ctk.CTkFont(size=16, weight="bold"), text_color=_THEME.text).pack(anchor="w", padx=14, pady=(12, 6))
        self.help_text = ctk.CTkTextbox(help_card, height=180, font=ctk.CTkFont(size=12), activate_scrollbars=True)
        self.help_text.pack(fill="x", padx=14, pady=(0, 14))
        set_widget_text(
            self.help_text,
            "鼠标：\n"
            "- 拖动 ROI 或参考区内部可移动\n"
            "- 拖动白色控制点可缩放\n"
            "- 双击空白区域可在该位置快速生成 ROI\n\n"
            "键盘：\n"
            "- A 新增参考区，Tab 轮换选中对象\n"
            "- R 选中 ROI\n"
            "- D/Delete 删除当前参考区\n"
            "- C 清空所有参考区\n"
            "- I/J/K/L 微调位置\n"
            "- Shift+I/J/K/L 微调尺寸\n"
            "- S 或 Enter 保存，Q 或 Esc 取消"
        )

        footer = ctk.CTkFrame(side, fg_color="transparent")
        footer.pack(fill="x", padx=18, pady=(0, 18))
        footer.grid_columnconfigure((0, 1), weight=1)
        self.status_label = ctk.CTkLabel(footer, text="状态：已同步", text_color=_THEME.success, anchor="w")
        self.status_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ctk.CTkButton(footer, text="取消", command=self.cancel, fg_color="transparent", border_width=1, border_color=_THEME.border, hover_color="#1e293b").grid(row=1, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(footer, text="保存到 YAML", command=self.save, fg_color=_THEME.accent, hover_color=_THEME.accent_hover).grid(row=1, column=1, padx=(6, 0), sticky="ew")

    def _on_tab_key(self, _event) -> str:
        self.cycle_selection()
        return "break"

    def _selected_rect(self) -> tuple[int, int, int, int] | None:
        if self.active_type == "roi":
            return self.roi
        if self.active_type == "ref" and 0 <= self.active_index < len(self.references):
            return self.references[self.active_index].rect
        return None

    def _set_selected_rect(self, rect: tuple[int, int, int, int]) -> None:
        clipped = fit_rect_to_bounds(rect, self.image_width, self.image_height, min_size=_MIN_REGION_SIZE)
        if self.active_type == "roi":
            self.roi = clipped
        elif self.active_type == "ref" and 0 <= self.active_index < len(self.references):
            self.references[self.active_index].rect = clipped
        else:
            return
        self.mark_dirty()
        self.refresh_view()

    def _render_frame(self) -> np.ndarray:
        frame = self.display_frame.copy()
        if self.roi is not None:
            draw_labeled_rect(
                frame,
                self.roi,
                self.display_scale,
                (34, 197, 94),
                "ROI",
                self.active_type == "roi",
            )
        for idx, ref in enumerate(self.references):
            selected = self.active_type == "ref" and idx == self.active_index
            color = (96, 165, 250) if selected else (245, 158, 11)
            draw_labeled_rect(frame, ref.rect, self.display_scale, color, ref.name, selected)
        return frame

    def refresh_view(self) -> None:
        self.canvas.set_image(self._render_frame())
        selected_text = "none"
        if self.active_type == "roi" and self.roi is not None:
            x, y, width, height = self.roi
            selected_text = f"ROI ({x}, {y}, {width}, {height})"
        elif self.active_type == "ref" and 0 <= self.active_index < len(self.references):
            ref = self.references[self.active_index]
            x, y, width, height = ref.rect
            selected_text = f"{ref.name} ({x}, {y}, {width}, {height})"

        roi_text = "未设置"
        if self.roi is not None:
            roi_text = f"x={self.roi[0]}, y={self.roi[1]}, w={self.roi[2]}, h={self.roi[3]}"
        self.meta_label.configure(
            text=(
                f"预览源：{self.preview.frame_source}\n"
                f"帧序号：{self.preview.frame_index}\n"
                f"原始尺寸：{self.image_width} × {self.image_height}\n"
                f"显示缩放：{self.display_scale:.3f}\n"
                f"ROI：{roi_text}\n"
                f"参考区数量：{len(self.references)}\n"
                f"当前选中：{selected_text}"
            )
        )

        if not self.references:
            refs_text = "暂无参考区。点击“新增参考区”或按 A 键创建。"
        else:
            lines = []
            for idx, ref in enumerate(self.references, start=1):
                x, y, width, height = ref.rect
                prefix = "* " if self.active_type == "ref" and self.active_index == idx - 1 else "  "
                lines.append(
                    f"{prefix}{idx:02d}. {ref.name:<10} x={x:<4d} y={y:<4d} w={width:<4d} h={height:<4d} weight={ref.weight:.2f}"
                )
            refs_text = "\n".join(lines)
        set_widget_text(self.refs_text, refs_text)
        self.status_label.configure(
            text="状态：已修改" if self.is_dirty else "状态：已同步",
            text_color=_THEME.warning if self.is_dirty else _THEME.success,
        )

    def mark_dirty(self) -> None:
        self.is_dirty = True

    def cycle_selection(self) -> None:
        order: list[tuple[str, int]] = []
        if self.roi is not None:
            order.append(("roi", -1))
        order.extend(("ref", idx) for idx in range(len(self.references)))
        if not order:
            self.active_type = "none"
            self.active_index = -1
            self.refresh_view()
            return
        current = (self.active_type, self.active_index)
        if current not in order:
            self.active_type, self.active_index = order[0]
        else:
            next_idx = (order.index(current) + 1) % len(order)
            self.active_type, self.active_index = order[next_idx]
        self.refresh_view()

    def select_roi(self) -> None:
        if self.roi is None:
            self.roi = _make_default_roi(self.image_width, self.image_height)
            self.mark_dirty()
        self.active_type = "roi"
        self.active_index = -1
        self.refresh_view()

    def add_reference(self) -> None:
        name = _generate_reference_name(self.references)
        rect = _make_default_reference(self.image_width, self.image_height, len(self.references))
        self.references.append(ReferenceRegionState(name=name, rect=rect, weight=1.0))
        self.active_type = "ref"
        self.active_index = len(self.references) - 1
        self.mark_dirty()
        self.refresh_view()

    def delete_selected_reference(self) -> None:
        if self.active_type != "ref" or not (0 <= self.active_index < len(self.references)):
            return
        del self.references[self.active_index]
        if self.references:
            self.active_index = min(self.active_index, len(self.references) - 1)
            self.active_type = "ref"
        else:
            self.active_type = "roi" if self.roi is not None else "none"
            self.active_index = -1
        self.mark_dirty()
        self.refresh_view()

    def clear_references(self) -> None:
        if not self.references:
            return
        self.references.clear()
        self.active_type = "roi" if self.roi is not None else "none"
        self.active_index = -1
        self.mark_dirty()
        self.refresh_view()

    def nudge_selected(self, dx: int, dy: int, resize: bool) -> None:
        rect = self._selected_rect()
        if rect is None:
            return
        x, y, width, height = rect
        updated = (x, y, width + dx, height + dy) if resize else (x + dx, y + dy, width, height)
        self._set_selected_rect(updated)

    def _handle_canvas_event(self, action: str, sx: int, sy: int) -> None:
        point = screen_to_image((sx, sy), self.display_scale, self.image_width, self.image_height)
        if action == "double":
            self.roi = fit_rect_to_bounds((point[0] - 80, point[1] - 80, 160, 160), self.image_width, self.image_height, min_size=_MIN_REGION_SIZE)
            self.active_type = "roi"
            self.active_index = -1
            self.mark_dirty()
            self.refresh_view()
            return

        if action == "press":
            if self.roi is not None:
                mode = hit_test_rect(self.roi, point, self.display_scale)
                if mode is not None:
                    self.active_type = "roi"
                    self.active_index = -1
                    self.drag_mode = mode
                    self.drag_start_point = point
                    self.drag_start_rect = self.roi
                    self.refresh_view()
                    return
            for idx in range(len(self.references) - 1, -1, -1):
                mode = hit_test_rect(self.references[idx].rect, point, self.display_scale)
                if mode is not None:
                    self.active_type = "ref"
                    self.active_index = idx
                    self.drag_mode = mode
                    self.drag_start_point = point
                    self.drag_start_rect = self.references[idx].rect
                    self.refresh_view()
                    return
            self.active_type = "none"
            self.active_index = -1
            self.refresh_view()
            return

        if action == "move":
            if self.drag_mode is None or self.drag_start_point is None or self.drag_start_rect is None:
                return
            updated = move_or_resize_rect(self.drag_start_rect, self.drag_mode, self.drag_start_point, point)
            self._set_selected_rect(updated)
            return

        if action == "release":
            self.drag_mode = None
            self.drag_start_point = None
            self.drag_start_rect = None

    def cancel(self) -> None:
        if self.is_dirty:
            if not messagebox.askyesno("取消配置", "当前修改尚未保存，确定要退出吗？", parent=self):
                return
        self.destroy()

    def save(self) -> None:
        if self.roi is None:
            messagebox.showerror("无法保存", "必须先定义 ROI。", parent=self)
            return
        try:
            _write_config_regions(self.config_path, self.roi, self.references)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)
            return

        reference_regions = [
            ReferenceRegion(
                name=ref.name,
                x=ref.rect[0],
                y=ref.rect[1],
                width=ref.rect[2],
                height=ref.rect[3],
                weight=ref.weight,
            )
            for ref in self.references
        ]
        self.result = ROIConfiguratorResult(
            frame_source=self.preview.frame_source,
            frame_index=self.preview.frame_index,
            roi=self.roi,
            reference_regions=reference_regions,
            config_path=self.config_path,
        )
        self.saved = True
        self.is_dirty = False
        self.destroy()


def configure_roi_interactively(
    config_path: str | Path,
    frame_index: int | None = None,
) -> ROIConfiguratorResult:
    path = Path(config_path)
    config = load_config(path)
    preview = _load_frame_preview(config, frame_index=frame_index)
    app = ROIConfiguratorApp(path, preview, config)
    app.mainloop()
    if not app.saved or app.result is None:
        raise ROIConfiguratorError("用户取消了 ROI/reference_regions 配置，未写入 YAML")
    return app.result
