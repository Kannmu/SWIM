from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import (
    CalibrationError,
    calibrate_from_config,
    capture_and_calibrate_from_config,
    collect_calibration_images_interactive,
)
from .capture import VideoCaptureSession
from .config import load_config, save_config_template
from .pipeline import run_analysis_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SWIM stroboscopic 2D DIC")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init-config", help="生成默认配置文件")
    init_cmd.add_argument("path", type=Path, help="输出 YAML 配置路径")

    capture_cmd = sub.add_parser("capture", help="录制频闪视频")
    capture_cmd.add_argument("config", type=Path, help="YAML 配置路径")
    capture_cmd.add_argument("--duration", type=float, default=2.0, help="录制时长，单位秒")

    analyze_cmd = sub.add_parser("analyze", help="分析已录制视频")
    analyze_cmd.add_argument("config", type=Path, help="YAML 配置路径")

    run_cmd = sub.add_parser("run", help="先录制再分析")
    run_cmd.add_argument("config", type=Path, help="YAML 配置路径")
    run_cmd.add_argument("--duration", type=float, default=2.0, help="录制时长，单位秒")

    calib_cmd = sub.add_parser("calibrate", help="使用棋盘格图像标定相机并写回配置")
    calib_cmd.add_argument("config", type=Path, help="YAML 配置路径")
    calib_cmd.add_argument("images_dir", type=Path, help="标定图像目录")
    calib_cmd.add_argument("--rows", type=int, default=9, help="棋盘格内角点行数")
    calib_cmd.add_argument("--cols", type=int, default=12, help="棋盘格内角点列数")
    calib_cmd.add_argument("--square-mm", type=float, default=15.0, help="单个方格物理尺寸，单位 mm")
    calib_cmd.add_argument("--output-json", type=Path, default=None, help="标定结果 JSON 输出路径")
    calib_cmd.add_argument("--no-write-config", action="store_true", help="仅计算标定，不写回 YAML 配置")

    calib_capture_cmd = sub.add_parser("capture-calibration", help="交互式采集棋盘格图像并保存")
    calib_capture_cmd.add_argument("config", type=Path, help="YAML 配置路径")
    calib_capture_cmd.add_argument("images_dir", type=Path, help="标定图像保存目录")
    calib_capture_cmd.add_argument("--rows", type=int, default=9, help="棋盘格内角点行数")
    calib_capture_cmd.add_argument("--cols", type=int, default=12, help="棋盘格内角点列数")
    calib_capture_cmd.add_argument("--square-mm", type=float, default=15.0, help="单个方格物理尺寸，单位 mm")
    calib_capture_cmd.add_argument("--min-score", type=float, default=70.0, help="允许保存图像的最低质量分数")

    calib_interactive_cmd = sub.add_parser("calibrate-interactive", help="交互式采集标定图像后立即完成标定")
    calib_interactive_cmd.add_argument("config", type=Path, help="YAML 配置路径")
    calib_interactive_cmd.add_argument("images_dir", type=Path, help="标定图像保存目录")
    calib_interactive_cmd.add_argument("--rows", type=int, default=9, help="棋盘格内角点行数")
    calib_interactive_cmd.add_argument("--cols", type=int, default=12, help="棋盘格内角点列数")
    calib_interactive_cmd.add_argument("--square-mm", type=float, default=15.0, help="单个方格物理尺寸，单位 mm")
    calib_interactive_cmd.add_argument("--output-json", type=Path, default=None, help="标定结果 JSON 输出路径")
    calib_interactive_cmd.add_argument("--no-write-config", action="store_true", help="仅计算标定，不写回 YAML 配置")
    calib_interactive_cmd.add_argument("--min-score", type=float, default=70.0, help="允许保存图像的最低质量分数")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-config":
        save_config_template(args.path)
        print(f"配置模板已写入: {args.path}")
        return

    if args.command == "capture-calibration":
        try:
            saved_paths = collect_calibration_images_interactive(
                config_path=args.config,
                output_dir=args.images_dir,
                inner_corners_rows=args.rows,
                inner_corners_cols=args.cols,
                square_size_mm=args.square_mm,
                min_score=args.min_score,
            )
        except CalibrationError as exc:
            parser.error(str(exc))
        print(f"交互式标定图像采集完成: saved_images={len(saved_paths)}, output_dir={args.images_dir}")
        return

    if args.command == "calibrate-interactive":
        try:
            saved_paths, result = capture_and_calibrate_from_config(
                config_path=args.config,
                images_dir=args.images_dir,
                inner_corners_rows=args.rows,
                inner_corners_cols=args.cols,
                square_size_mm=args.square_mm,
                output_json=args.output_json,
                write_config=not args.no_write_config,
                min_score=args.min_score,
            )
        except CalibrationError as exc:
            parser.error(str(exc))
        print(
            "交互式采集与标定完成: "
            f"saved_images={len(saved_paths)}, "
            f"pixel_size_um={result.mean_pixel_size_um:.6f}, "
            f"rms_reprojection_error_px={result.rms_reprojection_error_px:.6f}, "
            f"valid_images={result.valid_image_count}/{result.image_count}"
        )
        return

    config = load_config(args.config)

    if args.command == "capture":
        session = VideoCaptureSession(config)
        metadata = session.record(args.duration)
        print(f"视频采集完成: {metadata.video_path}")
        return

    if args.command == "analyze":
        run_analysis_pipeline(config)
        print(f"分析完成，输出目录: {config.paths.output_dir}")
        return

    if args.command == "run":
        session = VideoCaptureSession(config)
        session.record(args.duration)
        run_analysis_pipeline(config)
        print(f"录制与分析完成，输出目录: {config.paths.output_dir}")
        return

    if args.command == "calibrate":
        try:
            result = calibrate_from_config(
                config_path=args.config,
                images_dir=args.images_dir,
                inner_corners_rows=args.rows,
                inner_corners_cols=args.cols,
                square_size_mm=args.square_mm,
                output_json=args.output_json,
                write_config=not args.no_write_config,
            )
        except CalibrationError as exc:
            parser.error(str(exc))
        print(
            "标定完成: "
            f"pixel_size_um={result.mean_pixel_size_um:.6f}, "
            f"rms_reprojection_error_px={result.rms_reprojection_error_px:.6f}, "
            f"valid_images={result.valid_image_count}/{result.image_count}"
        )
        return
