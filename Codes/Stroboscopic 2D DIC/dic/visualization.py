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


def _save_xt_map(data: np.ndarray, positions_mm: np.ndarray, time_s: np.ndarray, title: str, path: Path, cmap: str, cbar: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.8))
    extent = [float(positions_mm[0]), float(positions_mm[-1]), float(time_s[-1]), float(time_s[0])]
    img = plt.imshow(data, aspect="auto", cmap=cmap, extent=extent)
    plt.xlabel("Position [mm]")
    plt.ylabel("Time [s]")
    plt.title(title)
    plt.colorbar(img, label=cbar)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _save_snapshot_grid(snapshots: np.ndarray, indices: np.ndarray, time_s: np.ndarray, path: Path, title_prefix: str, cmap: str, cbar: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = len(indices)
    fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 4.2), squeeze=False)
    vmax = float(np.nanmax(np.abs(snapshots))) if np.size(snapshots) else 1.0
    vmax = max(vmax, 1e-9)
    for ax, idx, snap in zip(axes[0], indices, snapshots):
        img = ax.imshow(snap, origin="lower", cmap=cmap, vmin=-vmax if cmap in {"coolwarm", "seismic", "RdBu_r"} else None, vmax=vmax if cmap in {"coolwarm", "seismic", "RdBu_r"} else None)
        ax.set_title(f"t={time_s[idx]:.3f} s")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title_prefix)
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.8, label=cbar)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_reference_region_overlay(pre: PreprocessResult, config: ProjectConfig, path: Path) -> None:
    frame = np.clip(pre.raw_frames[0], 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    offset_x, offset_y = pre.roi_offset_xy
    for region in config.reference_regions:
        x0 = int(region.x - offset_x)
        y0 = int(region.y - offset_y)
        x1 = int(x0 + region.width)
        y1 = int(y0 + region.height)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(rgb, region.name, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb)


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
    _save_heatmap(np.max(np.abs(stats.signed_wave_field), axis=0), "Peak signed wave magnitude [strain]", out / "signed_wave_peak.png", cmap="magma", cbar="strain")

    if config.reference_regions:
        _save_reference_region_overlay(pre, config, out / "reference_regions_overlay.png")

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
        stats.center_time_s,
        stats.center_strain_xy,
        "Center shear strain trace",
        "Time [s]",
        "Strain",
        out / "center_strain_xy_trace.png",
    )
    _save_line(
        stats.wave_profile_positions_mm,
        stats.wave_profile_amplitude_um,
        f"Wave amplitude profile along {stats.wave_profile_axis}",
        "Position [mm]",
        "Amplitude [um]",
        out / "wave_profile.png",
    )

    _save_xt_map(
        stats.xt_displacement,
        stats.wave_profile_positions_mm,
        stats.center_time_s,
        f"XT displacement map along {stats.wave_profile_axis}",
        out / "xt_displacement.png",
        cmap="magma",
        cbar="um",
    )
    _save_xt_map(
        stats.xt_strain_xy,
        stats.wave_profile_positions_mm,
        stats.center_time_s,
        f"XT shear strain map along {stats.wave_profile_axis}",
        out / "xt_strain_xy.png",
        cmap="RdBu_r",
        cbar="strain",
    )
    _save_snapshot_grid(
        stats.wavefront_displacement_snapshots,
        stats.wavefront_phase_indices,
        stats.center_time_s,
        out / "wavefront_displacement_snapshots.png",
        "Wavefront displacement snapshots",
        cmap="magma",
        cbar="um",
    )
    _save_snapshot_grid(
        stats.wavefront_strain_xy_snapshots,
        stats.wavefront_phase_indices,
        stats.center_time_s,
        out / "wavefront_strain_xy_snapshots.png",
        "Wavefront shear strain snapshots",
        cmap="RdBu_r",
        cbar="strain",
    )

    stride = max(1, config.analysis.visualize_quiver_stride)
    xs, ys = np.meshgrid(dic.grid.centers_x, dic.grid.centers_y)
    plt.figure(figsize=(8, 6))
    plt.imshow(stats.amp_total, origin="lower", cmap="magma", alpha=0.85)
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
    plt.savefig(out / "vector_field_overlay.png", dpi=220)
    plt.close()

    if config.analysis.export_field_csv:
        df = pd.DataFrame(
            {
                "time_s": stats.center_time_s,
                "center_displacement_um": stats.center_displacement_um,
                "ref_corrected_center_um": stats.ref_corrected_center_um,
                "center_strain_xy": stats.center_strain_xy,
                "reference_dx_px": pre.reference_region_motion[:, 0],
                "reference_dy_px": pre.reference_region_motion[:, 1],
                "common_mode_signal": pre.common_mode_signal,
            }
        )
        df.to_csv(out / "center_trace.csv", index=False)

        diag_df = pd.DataFrame(
            {
                "metric": [
                    "used_gpu",
                    "gpu_backend",
                    "batch_size",
                    "num_frames",
                    "grid_rows",
                    "grid_cols",
                    "num_points",
                    "total_matches",
                    "search_window_size_px",
                    "subset_size_px",
                    "search_radius_px",
                    "estimated_search_positions_per_point",
                    "estimated_fft_refinements",
                ],
                "value": [
                    dic.diagnostics.used_gpu,
                    dic.diagnostics.gpu_backend,
                    dic.diagnostics.batch_size,
                    dic.diagnostics.num_frames,
                    dic.diagnostics.grid_shape[0],
                    dic.diagnostics.grid_shape[1],
                    dic.diagnostics.num_points,
                    dic.diagnostics.total_matches,
                    dic.diagnostics.search_window_size_px,
                    dic.diagnostics.subset_size_px,
                    dic.diagnostics.search_radius_px,
                    dic.diagnostics.estimated_search_positions_per_point,
                    dic.diagnostics.estimated_fft_refinements,
                ],
            }
        )
        diag_df.to_csv(out / "runtime_diagnostics.csv", index=False)

    if config.analysis.export_video_overlays:
        _export_overlay_video(pre, dic, stats, out / "dic_overlay.mp4")
        _export_strain_video(pre, stats, out / "strain_overlay.mp4")


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


def _export_strain_video(pre: PreprocessResult, stats: FieldStatistics, path: Path) -> None:
    frames = pre.raw_frames
    h, w = frames.shape[1:]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h), True)
    if not writer.isOpened():
        return

    vmax = float(np.nanmax(np.abs(stats.signed_wave_field)))
    vmax = max(vmax, 1e-9)
    for t, frame in enumerate(frames):
        gray = np.clip(frame, 0, 255).astype(np.uint8)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if t < stats.signed_wave_field.shape[0]:
            field = np.clip(stats.signed_wave_field[t] / vmax, -1.0, 1.0)
            field_u8 = np.uint8(np.round((field + 1.0) * 127.5))
            color = cv2.applyColorMap(field_u8, cv2.COLORMAP_JET)
            color = cv2.resize(color, (w, h), interpolation=cv2.INTER_CUBIC)
            base = cv2.addWeighted(base, 0.55, color, 0.45, 0.0)
        text = f"frame={t:04d} shear={stats.center_strain_xy[min(t, len(stats.center_strain_xy)-1)]:.4e}"
        cv2.putText(base, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(base)
    writer.release()
