from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .types import DICResult, FieldStatistics, PreprocessResult


def _save_heatmap(data: np.ndarray, title: str, path: Path, cmap: str = "viridis", cbar: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    img = plt.imshow(data, origin="lower", cmap=cmap, aspect="auto")
    plt.title(title)
    plt.colorbar(img, label=cbar)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_line(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def export_visualizations(
    pre: PreprocessResult,
    dic: DICResult,
    stats: FieldStatistics,
    config: ProjectConfig,
) -> None:
    out = Path(config.paths.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _save_heatmap(pre.reference_frame, "Reference frame", out / "reference_frame.png", cmap="gray")
    _save_heatmap(stats.amp_total, "Total displacement amplitude [um]", out / "amplitude_total.png", cbar="um")
    _save_heatmap(stats.rms_u, "RMS horizontal displacement [um]", out / "rms_u.png", cbar="um")
    _save_heatmap(stats.rms_v, "RMS vertical displacement [um]", out / "rms_v.png", cbar="um")
    _save_heatmap(stats.temporal_std, "Temporal STD of displacement [um]", out / "temporal_std.png", cbar="um")
    _save_heatmap(stats.dominant_phase, "Dominant phase map [rad]", out / "phase_map.png", cmap="twilight", cbar="rad")
    _save_heatmap(stats.strain_xx, "Strain exx", out / "strain_xx.png", cmap="coolwarm")
    _save_heatmap(stats.strain_yy, "Strain eyy", out / "strain_yy.png", cmap="coolwarm")
    _save_heatmap(stats.strain_xy, "Strain exy", out / "strain_xy.png", cmap="coolwarm")

    _save_line(
        stats.center_time_s,
        stats.center_displacement_um,
        "Center displacement trace",
        "Time [s]",
        "Displacement [um]",
        out / "center_trace.png",
    )
    _save_line(
        stats.center_time_s,
        stats.ref_corrected_center_um,
        "Center displacement after reference correction",
        "Time [s]",
        "Displacement [um]",
        out / "center_trace_ref_corrected.png",
    )
    _save_line(
        stats.wave_profile_positions_mm,
        stats.wave_profile_amplitude_um,
        f"Wave amplitude profile along {stats.wave_profile_axis}",
        "Position [mm]",
        "Amplitude [um]",
        out / "wave_profile.png",
    )

    stride = max(1, config.analysis.visualize_quiver_stride)
    xs, ys = np.meshgrid(dic.grid.centers_x, dic.grid.centers_y)
    plt.figure(figsize=(8, 6))
    plt.imshow(stats.amp_total, origin="lower", cmap="magma", alpha=0.8)
    plt.quiver(
        xs[::stride, ::stride],
        ys[::stride, ::stride],
        stats.mean_u[::stride, ::stride],
        stats.mean_v[::stride, ::stride],
        color="cyan",
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.title("Mean displacement vector field over amplitude map")
    plt.tight_layout()
    plt.savefig(out / "vector_field_overlay.png", dpi=200)
    plt.close()

    if config.analysis.export_field_csv:
        df = pd.DataFrame(
            {
                "time_s": stats.center_time_s,
                "center_displacement_um": stats.center_displacement_um,
                "ref_corrected_center_um": stats.ref_corrected_center_um,
            }
        )
        df.to_csv(out / "center_trace.csv", index=False)

    if config.analysis.export_video_overlays:
        _export_overlay_video(pre, dic, stats, out / "dic_overlay.mp4")


def _export_overlay_video(pre: PreprocessResult, dic: DICResult, stats: FieldStatistics, path: Path) -> None:
    frames = pre.raw_frames
    h, w = frames.shape[1:]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h), True)
    if not writer.isOpened():
        return

    xs, ys = np.meshgrid(dic.grid.centers_x, dic.grid.centers_y)
    for t, frame in enumerate(frames):
        rgb = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if t < dic.u.shape[0]:
            for y, x, du, dv in zip(ys.ravel(), xs.ravel(), dic.u[t].ravel(), dic.v[t].ravel()):
                if np.isnan(du) or np.isnan(dv):
                    continue
                p0 = (int(x), int(y))
                p1 = (int(round(x + du)), int(round(y + dv)))
                cv2.arrowedLine(rgb, p0, p1, (0, 255, 255), 1, tipLength=0.25)
        text = f"frame={t:04d} center={stats.ref_corrected_center_um[min(t, len(stats.ref_corrected_center_um)-1)]:.3f} um"
        cv2.putText(rgb, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(rgb)
    writer.release()
