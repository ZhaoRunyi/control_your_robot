#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calib.dual_piper_arm_extrinsic_calib import (
    LiveRobotSnapshotReader,
    PiperDualCalibrationRobot,
    TerminalCommandReader,
    controller_name_for_arm,
    generate_sample_all_raw,
    load_saved_run,
    matrix_section,
    other_arm_name,
    pose_delta_metrics,
    prompt_for_arm,
    resolve_saved_run_dir,
    resize_for_preview,
    save_sample,
    save_samples_json,
    write_text,
)
from calib.utils import (
    DEFAULT_TAG_DICTIONARY,
    DEFAULT_TAG_ID,
    DEFAULT_TAG_SIZE_MM,
    arm_to_can_bus,
    detect_tag_pose,
    ensure_dir,
    format_matrix,
    format_matrix_list,
    get_realsense_intrinsics,
    matrix_to_rounded_nested_list,
    overlay_status_panel,
    run_eye_in_hand_calibration,
    save_json,
    timestamp_for_path,
)


WINDOW_NAME = "Dual Piper Eye-In-Hand Calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual Piper wrist camera eye-in-hand calibration")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help=(
            "existing run directory to recompute in offline mode; "
            "accepts an absolute path, a cwd-relative path, or a bare run directory name under calib/runs"
        ),
    )
    parser.add_argument(
        "--offline-run-dir",
        type=str,
        default=None,
        help=(
            "deprecated alias for the positional run_dir argument; "
            "skip robot/camera IO and recompute calibration from an existing run directory"
        ),
    )
    parser.add_argument("--left_cam_serial", type=str, default="344322073012")
    parser.add_argument("--high_cam_serial", type=str, default="323422071854")
    parser.add_argument("--right_cam_serial", type=str, default="335522070790")
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="display resize scale for the interactive preview window",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="discard the first N frames before starting calibration",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="decimal places for the copyable matrix text output",
    )
    return parser.parse_args()


def wrist_camera_name_for_arm(arm_name: str) -> str:
    if arm_name == "left_arm":
        return "left"
    if arm_name == "right_arm":
        return "right"
    raise ValueError(f"unsupported arm name: {arm_name}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(inner_value) for inner_value in value]
    return value


def build_preview_lines(
    arm_name: str,
    camera_name: str,
    sample_count: int,
    detection_found: bool,
    ee_pose: np.ndarray,
    selected_delta_text: str,
    other_delta_text: str,
    last_status: str,
) -> list[str]:
    return [
        f"setup=eye_in_hand  arm={arm_name}  camera=cam_{camera_name}  samples={sample_count}",
        f"tag={'FOUND' if detection_found else 'NOT FOUND'}  static=base  dict={DEFAULT_TAG_DICTIONARY}  id={DEFAULT_TAG_ID}",
        f"ee_pose=[{', '.join(f'{value:.4f}' for value in ee_pose)}]",
        selected_delta_text,
        other_delta_text,
        "terminal only: c<Enter> save | q<Enter> quit",
        last_status,
    ]


def compute_base_camera_per_sample(
    samples: list[dict[str, Any]],
    gripper_to_camera_transform: np.ndarray,
) -> list[np.ndarray]:
    return [
        np.asarray(sample["T_base_gripper"], dtype=np.float64) @ gripper_to_camera_transform
        for sample in samples
    ]


def compute_reference_base_camera(
    samples: list[dict[str, Any]],
    gripper_to_camera_transform: np.ndarray,
    reference_sample_index: int = 0,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot compute reference base-camera transform from an empty sample list")
    if reference_sample_index < 0 or reference_sample_index >= len(samples):
        raise IndexError(
            f"reference_sample_index={reference_sample_index} is out of range for {len(samples)} samples"
        )
    reference_sample = samples[reference_sample_index]
    base_to_gripper = np.asarray(reference_sample["T_base_gripper"], dtype=np.float64)
    base_to_camera = base_to_gripper @ gripper_to_camera_transform
    return {
        "reference_sample_index": int(reference_sample_index),
        "reference_ee_pose": np.asarray(reference_sample["ee_pose"], dtype=np.float64).reshape(6),
        "T_base_camera": base_to_camera,
        "T_camera_base": np.linalg.inv(base_to_camera),
        "T_base_camera_per_sample": compute_base_camera_per_sample(samples, gripper_to_camera_transform),
    }


def build_report_text(
    run_name: str,
    arm_name: str,
    camera_name: str,
    intrinsics: dict[str, Any],
    samples: list[dict[str, Any]],
    calibration_result: dict[str, Any],
    decimals: int,
) -> str:
    best_method = calibration_result["best_method"]
    best_result = calibration_result["best_result"]
    all_results = calibration_result["all_results"]
    reference_result = compute_reference_base_camera(samples, best_result["T_gripper_camera"])

    lines = [
        f"run_name: {run_name}",
        "calibration_mode: eye_in_hand",
        f"arm_name: {arm_name}",
        f"camera_name: cam_{camera_name}",
        f"sample_count: {len(samples)}",
        f"selected_method: {best_method}",
        f"reference_sample_index: {reference_result['reference_sample_index']}",
        f"reference_ee_pose: {[round(float(value), 6) for value in reference_result['reference_ee_pose']]}",
        f"default_tag_dictionary: {DEFAULT_TAG_DICTIONARY}",
        f"default_tag_id: {DEFAULT_TAG_ID}",
        f"default_tag_size_mm: {DEFAULT_TAG_SIZE_MM:.1f}",
        "",
        "camera_intrinsics:",
        format_matrix(np.asarray(intrinsics["camera_matrix"], dtype=np.float64), decimals=decimals),
        "",
        "best_metrics:",
        f"- reprojection_error_mean_px: {best_result['reprojection_error_mean_px']:.4f}",
        f"- reprojection_error_rmse_px: {best_result['reprojection_error_rmse_px']:.4f}",
        f"- reprojection_error_max_px: {best_result['reprojection_error_max_px']:.4f}",
        f"- tag_in_base_translation_rmse_mm: {best_result['tag_in_base_translation_rmse_mm']:.4f}",
        f"- tag_in_base_translation_max_mm: {best_result['tag_in_base_translation_max_mm']:.4f}",
        f"- tag_in_base_rotation_rmse_deg: {best_result['tag_in_base_rotation_rmse_deg']:.4f}",
        f"- tag_in_base_rotation_max_deg: {best_result['tag_in_base_rotation_max_deg']:.4f}",
        f"- camera_in_gripper_translation_rmse_mm: {best_result['camera_in_gripper_translation_rmse_mm']:.4f}",
        f"- camera_in_gripper_translation_max_mm: {best_result['camera_in_gripper_translation_max_mm']:.4f}",
        f"- camera_in_gripper_rotation_rmse_deg: {best_result['camera_in_gripper_rotation_rmse_deg']:.4f}",
        f"- camera_in_gripper_rotation_max_deg: {best_result['camera_in_gripper_rotation_max_deg']:.4f}",
        f"- arm_translation_span_xyz_mm: {[round(value, 4) for value in best_result['motion_summary']['translation_span_xyz_mm']]}",
        f"- arm_translation_span_norm_mm: {best_result['motion_summary']['translation_span_norm_mm']:.4f}",
        f"- arm_max_relative_rotation_deg: {best_result['motion_summary']['max_relative_rotation_deg']:.4f}",
        "",
        matrix_section(
            "T_base_camera (camera pose in arm base frame, evaluated at reference sample):",
            reference_result["T_base_camera"],
            decimals,
        ),
        "",
        matrix_section(
            "T_camera_base (arm base pose in camera frame, evaluated at reference sample):",
            reference_result["T_camera_base"],
            decimals,
        ),
        "",
        "sample_T_base_gripper_list:",
        format_matrix_list([sample["T_base_gripper"] for sample in samples], decimals=decimals),
        "",
        "sample_T_camera_tag_list:",
        format_matrix_list([sample["T_camera_tag"] for sample in samples], decimals=decimals),
        "",
        "sample_T_base_camera_list:",
        format_matrix_list(reference_result["T_base_camera_per_sample"], decimals=decimals),
        "",
        "supplemental_eye_in_hand_transforms:",
        matrix_section("T_gripper_camera (camera pose in end-effector frame):", best_result["T_gripper_camera"], decimals),
        "",
        matrix_section("T_camera_gripper (end-effector pose in camera frame):", best_result["T_camera_gripper"], decimals),
        "",
        matrix_section("T_base_tag (tag pose in arm base frame):", best_result["T_base_tag"], decimals),
        "",
        matrix_section("T_tag_base (arm base pose in tag frame):", best_result["T_tag_base"], decimals),
        "",
        "all_method_metrics:",
    ]

    for method_name, method_result in all_results.items():
        if not method_result.get("success"):
            lines.append(f"- {method_name}: failed: {method_result.get('error', 'unknown error')}")
            continue
        lines.extend(
            [
                f"- {method_name}:",
                f"  reproj_rmse_px={method_result['reprojection_error_rmse_px']:.4f}, "
                f"cam_in_gripper_trans_rmse_mm={method_result['camera_in_gripper_translation_rmse_mm']:.4f}, "
                f"cam_in_gripper_rot_rmse_deg={method_result['camera_in_gripper_rotation_rmse_deg']:.4f}, "
                f"tag_in_base_trans_rmse_mm={method_result['tag_in_base_translation_rmse_mm']:.4f}, "
                f"tag_in_base_rot_rmse_deg={method_result['tag_in_base_rotation_rmse_deg']:.4f}",
            ]
        )
    return "\n".join(lines) + "\n"


def save_calibration_outputs(
    run_dir: Path,
    run_name: str,
    arm_name: str,
    camera_name: str,
    intrinsics: dict[str, Any],
    samples: list[dict[str, Any]],
    calibration_result: dict[str, Any],
    decimals: int,
) -> None:
    best_method = calibration_result["best_method"]
    best_result = calibration_result["best_result"]
    all_results = calibration_result["all_results"]
    reference_result = compute_reference_base_camera(samples, best_result["T_gripper_camera"])

    report_text = build_report_text(
        run_name=run_name,
        arm_name=arm_name,
        camera_name=camera_name,
        intrinsics=intrinsics,
        samples=samples,
        calibration_result=calibration_result,
        decimals=decimals,
    )
    write_text(run_dir / "calibration_report.txt", report_text)
    write_text(run_dir / "copyable_matrices.txt", report_text)

    json_payload = {
        "run_name": run_name,
        "calibration_mode": "eye_in_hand",
        "arm_name": arm_name,
        "camera_name": f"cam_{camera_name}",
        "sample_count": len(samples),
        "selected_method": best_method,
        "reference_sample_index": reference_result["reference_sample_index"],
        "reference_ee_pose": [float(value) for value in reference_result["reference_ee_pose"]],
        "default_tag": {
            "dictionary_name": DEFAULT_TAG_DICTIONARY,
            "marker_id": DEFAULT_TAG_ID,
            "tag_size_mm": float(DEFAULT_TAG_SIZE_MM),
        },
        "camera_intrinsics": {
            "width": intrinsics["width"],
            "height": intrinsics["height"],
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "ppx": intrinsics["ppx"],
            "ppy": intrinsics["ppy"],
            "distortion_model": intrinsics["distortion_model"],
            "coeffs": intrinsics["coeffs"],
            "camera_matrix": matrix_to_rounded_nested_list(intrinsics["camera_matrix"], decimals=decimals),
        },
        "best_metrics": {
            key: to_jsonable(value)
            for key, value in best_result.items()
            if key
            not in {
                "T_gripper_camera",
                "T_camera_gripper",
                "T_base_tag",
                "T_tag_base",
                "T_gripper_camera_per_sample",
                "success",
                "selection_score",
            }
        },
        "T_base_camera": matrix_to_rounded_nested_list(reference_result["T_base_camera"], decimals=decimals),
        "T_camera_base": matrix_to_rounded_nested_list(reference_result["T_camera_base"], decimals=decimals),
        "sample_T_base_camera_list": [
            matrix_to_rounded_nested_list(matrix, decimals=decimals)
            for matrix in reference_result["T_base_camera_per_sample"]
        ],
        "supplemental_eye_in_hand": {
            "T_gripper_camera": matrix_to_rounded_nested_list(best_result["T_gripper_camera"], decimals=decimals),
            "T_camera_gripper": matrix_to_rounded_nested_list(best_result["T_camera_gripper"], decimals=decimals),
            "T_base_tag": matrix_to_rounded_nested_list(best_result["T_base_tag"], decimals=decimals),
            "T_tag_base": matrix_to_rounded_nested_list(best_result["T_tag_base"], decimals=decimals),
        },
        "all_method_metrics": {},
    }
    for method_name, method_result in all_results.items():
        if not method_result.get("success"):
            json_payload["all_method_metrics"][method_name] = {
                "success": False,
                "error": method_result.get("error", "unknown error"),
            }
            continue
        json_payload["all_method_metrics"][method_name] = {
            "success": True,
            "reprojection_error_mean_px": method_result["reprojection_error_mean_px"],
            "reprojection_error_rmse_px": method_result["reprojection_error_rmse_px"],
            "reprojection_error_max_px": method_result["reprojection_error_max_px"],
            "camera_in_gripper_translation_rmse_mm": method_result["camera_in_gripper_translation_rmse_mm"],
            "camera_in_gripper_translation_max_mm": method_result["camera_in_gripper_translation_max_mm"],
            "camera_in_gripper_rotation_rmse_deg": method_result["camera_in_gripper_rotation_rmse_deg"],
            "camera_in_gripper_rotation_max_deg": method_result["camera_in_gripper_rotation_max_deg"],
            "tag_in_base_translation_rmse_mm": method_result["tag_in_base_translation_rmse_mm"],
            "tag_in_base_translation_max_mm": method_result["tag_in_base_translation_max_mm"],
            "tag_in_base_rotation_rmse_deg": method_result["tag_in_base_rotation_rmse_deg"],
            "tag_in_base_rotation_max_deg": method_result["tag_in_base_rotation_max_deg"],
        }
    save_json(run_dir / "calibration_result.json", to_jsonable(json_payload))


def finalize_calibration_run(
    run_dir: Path,
    run_name: str,
    arm_name: str,
    camera_name: str,
    intrinsics: dict[str, Any],
    samples: list[dict[str, Any]],
    decimals: int,
) -> int:
    overlay_path = run_dir / "sample_all_raw.png"
    if not overlay_path.is_file():
        generate_sample_all_raw(run_dir, samples)

    calibration_result = run_eye_in_hand_calibration(samples)
    save_calibration_outputs(
        run_dir=run_dir,
        run_name=run_name,
        arm_name=arm_name,
        camera_name=camera_name,
        intrinsics=intrinsics,
        samples=samples,
        calibration_result=calibration_result,
        decimals=decimals,
    )

    best_result = calibration_result["best_result"]
    reference_result = compute_reference_base_camera(samples, best_result["T_gripper_camera"])
    print("")
    print(f"selected_method: {calibration_result['best_method']}")
    print("T_base_camera:")
    print(format_matrix(reference_result["T_base_camera"], decimals=decimals))
    print("")
    print("T_camera_base:")
    print(format_matrix(reference_result["T_camera_base"], decimals=decimals))
    print("")
    print(
        "metrics: "
        f"reproj_rmse_px={best_result['reprojection_error_rmse_px']:.4f}, "
        f"tag_in_base_trans_rmse_mm={best_result['tag_in_base_translation_rmse_mm']:.4f}, "
        f"tag_in_base_rot_rmse_deg={best_result['tag_in_base_rotation_rmse_deg']:.4f}, "
        f"cam_in_gripper_trans_rmse_mm={best_result['camera_in_gripper_translation_rmse_mm']:.4f}, "
        f"cam_in_gripper_rot_rmse_deg={best_result['camera_in_gripper_rotation_rmse_deg']:.4f}"
    )
    print(
        f"reference_sample_index: {reference_result['reference_sample_index']} "
        f"(camera pose evaluated from that sample's ee_pose)"
    )
    print(f"report_path: {run_dir / 'calibration_report.txt'}")
    return 0


def run_offline_calibration(args: argparse.Namespace) -> int:
    run_dir = resolve_saved_run_dir(args.offline_run_dir)
    run_name, arm_name, camera_name, intrinsics, samples = load_saved_run(run_dir)
    expected_camera_name = wrist_camera_name_for_arm(arm_name)
    if camera_name != expected_camera_name:
        raise ValueError(
            f"run {run_name} is not a wrist-camera eye-in-hand run: "
            f"arm_name={arm_name}, expected camera=cam_{expected_camera_name}, got cam_{camera_name}"
        )

    print(f"offline_run_dir: {run_dir}")
    print("calibration_mode: eye_in_hand")
    print(f"run_name: {run_name}")
    print(f"arm_name: {arm_name}")
    print(f"camera: cam_{camera_name}")
    print(f"samples_path: {run_dir / 'samples.json'}")
    print(f"intrinsics_path: {run_dir / 'camera_intrinsics.json'}")
    print(f"sample_count: {len(samples)}")

    overlay_path = run_dir / "sample_all_raw.png"
    if not overlay_path.is_file():
        overlay_path = generate_sample_all_raw(run_dir, samples)
    if overlay_path is not None and overlay_path.is_file():
        print(f"sample_all_raw_path: {overlay_path}")

    if len(samples) < 3:
        error_text = (
            f"有效样本不足，当前仅 {len(samples)} 个。"
            "至少需要 3 个不同位姿样本才能计算外参。"
        )
        write_text(run_dir / "calibration_report.txt", error_text + "\n")
        print(error_text)
        return 1

    return finalize_calibration_run(
        run_dir=run_dir,
        run_name=run_name,
        arm_name=arm_name,
        camera_name=camera_name,
        intrinsics=intrinsics,
        samples=samples,
        decimals=args.decimals,
    )


def main() -> int:
    args = parse_args()
    offline_run_dir = args.offline_run_dir or args.run_dir
    if offline_run_dir:
        args.offline_run_dir = offline_run_dir
        return run_offline_calibration(args)

    arm_name = prompt_for_arm()
    camera_name = wrist_camera_name_for_arm(arm_name)
    run_name = f"{timestamp_for_path(datetime.now())}_{arm_name}_eye_in_hand"

    calib_root = ensure_dir(PROJECT_ROOT / "calib")
    runs_root = ensure_dir(calib_root / "runs")
    run_dir = ensure_dir(runs_root / run_name)

    condition = {
        "left_cam_serial": args.left_cam_serial,
        "high_cam_serial": args.high_cam_serial,
        "right_cam_serial": args.right_cam_serial,
    }

    robot = PiperDualCalibrationRobot(condition=condition, arm_name=arm_name, camera_name=camera_name)
    command_reader = TerminalCommandReader()
    snapshot_reader: LiveRobotSnapshotReader | None = None
    samples: list[dict[str, Any]] = []
    intrinsics: dict[str, Any] | None = None
    reference_arm_poses: dict[str, np.ndarray] | None = None
    last_status = "终端输入 c 回车采样，输入 q 回车结束并计算。"

    print(f"run_dir: {run_dir}")
    print("calibration_mode: eye_in_hand")
    print(f"camera: cam_{camera_name}")
    print(
        "default_tag: "
        f"{DEFAULT_TAG_DICTIONARY}, id={DEFAULT_TAG_ID}, size={DEFAULT_TAG_SIZE_MM:.1f}mm"
    )
    print(
        "mapping: "
        f"l -> left_arm -> {controller_name_for_arm('left_arm')} -> {arm_to_can_bus('left_arm')} -> cam_left, "
        f"r -> right_arm -> {controller_name_for_arm('right_arm')} -> {arm_to_can_bus('right_arm')} -> cam_right"
    )
    print(
        "selected: "
        f"{arm_name} -> {controller_name_for_arm(arm_name)} -> {arm_to_can_bus(arm_name)} -> cam_{camera_name}"
    )
    print("假设: 相机固定在所选末端上，tag 固定在世界/底座环境中。")
    print("交互方式: 只使用终端。输入 c 后回车保存，输入 q 后回车结束。")
    print("窗口只用于预览，不处理键盘事件。")

    try:
        robot.set_up()
        command_reader.start()
        camera_key = f"cam_{camera_name}"
        intrinsics = get_realsense_intrinsics(robot.sensors["image"][camera_key])
        snapshot_reader = LiveRobotSnapshotReader(robot=robot, camera_key=camera_key)
        snapshot_reader.start()
        if not snapshot_reader.wait_until_ready(timeout_sec=5.0):
            raise RuntimeError("timed out waiting for the first robot snapshot")
        while snapshot_reader.frame_count() < max(args.warmup_frames, 1):
            _, snapshot_error = snapshot_reader.get_latest()
            if snapshot_error is not None:
                raise snapshot_error
            time.sleep(0.02)

        intrinsics_json = {
            "width": intrinsics["width"],
            "height": intrinsics["height"],
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "ppx": intrinsics["ppx"],
            "ppy": intrinsics["ppy"],
            "distortion_model": intrinsics["distortion_model"],
            "coeffs": intrinsics["coeffs"],
            "camera_matrix": matrix_to_rounded_nested_list(intrinsics["camera_matrix"], decimals=args.decimals),
        }
        save_json(run_dir / "camera_intrinsics.json", intrinsics_json)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 800, 600)
        while True:
            snapshot, snapshot_error = snapshot_reader.get_latest()
            if snapshot_error is not None:
                raise snapshot_error
            if snapshot is None:
                time.sleep(0.01)
                continue

            controller_data = snapshot["controller_data"]
            arm_data = controller_data[arm_name]
            image_rgb = np.asarray(snapshot["image_rgb"], dtype=np.uint8)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            arm_ee_pose_snapshot = {
                arm_key: np.asarray(arm_state["ee_pose"], dtype=np.float64).reshape(6)
                for arm_key, arm_state in controller_data.items()
            }
            if reference_arm_poses is None:
                reference_arm_poses = {
                    arm_key: arm_pose.copy() for arm_key, arm_pose in arm_ee_pose_snapshot.items()
                }

            detection, annotated_bgr = detect_tag_pose(
                image_bgr=image_bgr,
                camera_matrix=np.asarray(intrinsics["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.asarray(intrinsics["dist_coeffs"], dtype=np.float64),
                dictionary_name=DEFAULT_TAG_DICTIONARY,
                marker_id=DEFAULT_TAG_ID,
                tag_size_m=DEFAULT_TAG_SIZE_MM / 1000.0,
            )

            ee_pose = np.asarray(arm_data["ee_pose"], dtype=np.float64).reshape(6)
            selected_delta_mm, selected_delta_deg = pose_delta_metrics(reference_arm_poses[arm_name], ee_pose)
            other_name = other_arm_name(arm_name)
            other_delta_mm, other_delta_deg = pose_delta_metrics(
                reference_arm_poses[other_name],
                arm_ee_pose_snapshot[other_name],
            )
            preview_lines = build_preview_lines(
                arm_name=arm_name,
                camera_name=camera_name,
                sample_count=len(samples),
                detection_found=detection is not None,
                ee_pose=ee_pose,
                selected_delta_text=(
                    f"selected_delta[{arm_name}]=({selected_delta_mm:.1f}mm, {selected_delta_deg:.1f}deg)"
                ),
                other_delta_text=(
                    f"other_delta[{other_name}]=({other_delta_mm:.1f}mm, {other_delta_deg:.1f}deg)"
                ),
                last_status=last_status,
            )
            preview_bgr_fullres = overlay_status_panel(annotated_bgr, preview_lines)
            cv2.imshow(WINDOW_NAME, resize_for_preview(preview_bgr_fullres, args.preview_scale))

            cv2.waitKey(1)
            terminal_commands = command_reader.poll_all()
            save_requested = any(command in {"c", "save", "s"} for command in terminal_commands)
            quit_requested = any(command in {"q", "quit", "exit"} for command in terminal_commands)

            if save_requested:
                if detection is None:
                    last_status = "当前帧未检测到指定 tag，未保存。"
                    print(last_status)
                    continue
                sample_index = len(samples)
                saved_preview_bgr = overlay_status_panel(
                    annotated_bgr,
                    build_preview_lines(
                        arm_name=arm_name,
                        camera_name=camera_name,
                        sample_count=sample_index + 1,
                        detection_found=True,
                        ee_pose=ee_pose,
                        selected_delta_text=(
                            f"selected_delta[{arm_name}]=({selected_delta_mm:.1f}mm, {selected_delta_deg:.1f}deg)"
                        ),
                        other_delta_text=(
                            f"other_delta[{other_name}]=({other_delta_mm:.1f}mm, {other_delta_deg:.1f}deg)"
                        ),
                        last_status=f"sample_{sample_index:03d} saved",
                    ),
                )
                sample = save_sample(
                    run_dir=run_dir,
                    sample_index=sample_index,
                    arm_name=arm_name,
                    camera_name=camera_name,
                    ee_pose=ee_pose,
                    arm_ee_pose_snapshot=arm_ee_pose_snapshot,
                    detection=detection,
                    image_bgr=image_bgr,
                    annotated_bgr=annotated_bgr,
                    preview_bgr_fullres=saved_preview_bgr,
                )
                samples.append(sample)
                save_samples_json(run_dir, samples, decimals=args.decimals)
                overlay_path = generate_sample_all_raw(run_dir, samples)
                save_status = (
                    f"已保存 sample_{sample_index:03d} "
                    f"(reproj={detection.reprojection_error_px:.3f}px)"
                )
                if overlay_path is not None:
                    save_status += f" | sample_all_raw={overlay_path.name}"
                if (
                    selected_delta_mm < 1.0
                    and selected_delta_deg < 1.0
                    and (other_delta_mm > 5.0 or other_delta_deg > 5.0)
                ):
                    save_status += (
                        f" | 警告: {arm_name} 基本未动，但 {other_name} 在动。"
                        "请检查 l/r 选择、主从映射或 CAN 接线。"
                    )
                last_status = save_status
                print(last_status)
            elif quit_requested:
                break

        if len(samples) < 3:
            error_text = (
                f"有效样本不足，当前仅 {len(samples)} 个。"
                "至少需要 3 个不同位姿样本才能计算外参。"
            )
            write_text(run_dir / "calibration_report.txt", error_text + "\n")
            print(error_text)
            return 1

        return finalize_calibration_run(
            run_dir=run_dir,
            run_name=run_name,
            arm_name=arm_name,
            camera_name=camera_name,
            intrinsics=intrinsics,
            samples=samples,
            decimals=args.decimals,
        )
    finally:
        command_reader.stop()
        if snapshot_reader is not None:
            snapshot_reader.stop()
        cv2.destroyAllWindows()
        robot.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
