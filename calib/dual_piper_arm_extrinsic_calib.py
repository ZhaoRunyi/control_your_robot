#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot.controller.Piper_controller import PiperController
from robot.robot.base_robot import Robot
from robot.sensor.Realsense_sensor import RealsenseSensor

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
    normalize_arm_label,
    overlay_status_panel,
    pose6d_to_matrix,
    run_hand_eye_calibration,
    save_json,
    timestamp_for_path,
)


WINDOW_NAME = "Dual Piper Extrinsic Calibration"


class PiperDualCalibrationRobot(Robot):
    def __init__(self, condition: dict[str, Any], arm_name: str, camera_name: str):
        super().__init__(condition=condition, move_check=False, start_episode=0)
        self.name = "piper_dual_calibration_robot"
        self.arm_name = arm_name
        self.camera_name = camera_name

        self.left_cam = condition.get("left_cam_serial")
        self.high_cam = condition.get("high_cam_serial")
        self.right_cam = condition.get("right_cam_serial")

        self.controllers = {
            "arm": {
                "left_arm": PiperController("slave_left_arm"),
                "right_arm": PiperController("slave_right_arm"),
            }
        }
        self.sensors = {"image": {}}
        self.sensors["image"][f"cam_{camera_name}"] = RealsenseSensor(f"cam_{camera_name}")

    def set_up(self) -> None:
        super().set_up()
        # Align with dual_piper_arm_teleop.py: always initialize both slave arms.
        self.controllers["arm"]["left_arm"].set_up("can0")
        self.controllers["arm"]["right_arm"].set_up("can1")

        serial_map = {
            "high": self.high_cam,
            "left": self.left_cam,
            "right": self.right_cam,
        }
        camera_serial = serial_map[self.camera_name]
        if not camera_serial:
            raise ValueError(f"missing serial number for cam_{self.camera_name}")
        self.sensors["image"][f"cam_{self.camera_name}"].set_up(camera_serial)
        self.set_collect_type({"arm": ["joint", "ee_pose", "gripper"], "image": ["color"]})

    def cleanup(self) -> None:
        for sensor_group in self.sensors.values():
            for sensor in sensor_group.values():
                if hasattr(sensor, "cleanup"):
                    sensor.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual Piper external camera extrinsic calibration")
    parser.add_argument(
        "--offline-run-dir",
        type=str,
        default=None,
        help=(
            "skip robot/camera IO and recompute calibration from an existing run directory; "
            "accepts an absolute path, a cwd-relative path, or a bare run directory name under calib/runs"
        ),
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="high",
        choices=["high", "left", "right"],
        help="which camera to use for calibration",
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


def prompt_for_arm() -> str:
    while True:
        arm_text = input("请输入标定机械臂 [l/r]: ").strip()
        try:
            return normalize_arm_label(arm_text)
        except ValueError:
            print("输入无效，只接受 l 或 r。")


def resize_for_preview(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-8:
        return image_bgr
    height, width = image_bgr.shape[:2]
    return cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def resolve_saved_run_dir(run_dir_arg: str) -> Path:
    candidate = Path(run_dir_arg).expanduser()
    search_paths: list[Path] = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.append((Path.cwd() / candidate).resolve())
        search_paths.append((PROJECT_ROOT / "calib" / "runs" / candidate).resolve())

    for path in search_paths:
        if path.is_dir():
            return path
    checked = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"run directory not found; checked: {checked}")


def normalize_loaded_sample(sample: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(sample)
    if "T_base_gripper" not in normalized:
        if "ee_pose" not in normalized:
            raise KeyError("sample is missing both T_base_gripper and ee_pose")
        normalized["T_base_gripper"] = pose6d_to_matrix(normalized["ee_pose"])
    if "T_camera_tag" not in normalized:
        raise KeyError("sample is missing T_camera_tag")
    return normalized


def load_saved_run(run_dir: Path) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]]:
    samples_path = run_dir / "samples.json"
    intrinsics_path = run_dir / "camera_intrinsics.json"
    if not samples_path.is_file():
        raise FileNotFoundError(f"missing samples file: {samples_path}")
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"missing intrinsics file: {intrinsics_path}")

    samples_payload = json.loads(samples_path.read_text(encoding="utf-8"))
    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    raw_samples = samples_payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"no samples found in {samples_path}")

    samples = [normalize_loaded_sample(sample) for sample in raw_samples]
    arm_names = {str(sample["arm_name"]) for sample in samples}
    if len(arm_names) != 1:
        raise ValueError(f"inconsistent arm_name values in {samples_path}: {sorted(arm_names)}")
    camera_names = {str(sample["camera_name"]) for sample in samples}
    if len(camera_names) != 1:
        raise ValueError(f"inconsistent camera_name values in {samples_path}: {sorted(camera_names)}")

    arm_name = next(iter(arm_names))
    camera_name = next(iter(camera_names))
    if camera_name.startswith("cam_"):
        camera_name = camera_name[4:]
    return run_dir.name, arm_name, camera_name, intrinsics, samples


def collect_sample_raw_paths(run_dir: Path, samples: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for sample in samples:
        raw_path_value = sample.get("image_raw_path")
        if isinstance(raw_path_value, str) and raw_path_value:
            raw_path = Path(raw_path_value).expanduser()
            if not raw_path.is_absolute():
                raw_path = (run_dir / raw_path).resolve()
        else:
            sample_index = int(sample["sample_index"])
            raw_path = run_dir / f"sample_{sample_index:03d}_raw.png"
        raw_path = raw_path.resolve()
        if raw_path.is_file() and raw_path not in seen:
            paths.append(raw_path)
            seen.add(raw_path)
    return paths


def generate_sample_all_raw(run_dir: Path, samples: list[dict[str, Any]]) -> Path | None:
    raw_paths = collect_sample_raw_paths(run_dir, samples)
    if not raw_paths:
        return None

    overlay_accumulator: np.ndarray | None = None
    image_shape: tuple[int, ...] | None = None
    for index, raw_path in enumerate(raw_paths, start=1):
        image_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"failed to read raw image: {raw_path}")
        if image_shape is None:
            image_shape = image_bgr.shape
            overlay_accumulator = image_bgr.astype(np.float32)
            continue
        if image_bgr.shape != image_shape:
            raise ValueError(
                f"raw image shape mismatch: expected {image_shape}, got {image_bgr.shape} for {raw_path}"
            )
        assert overlay_accumulator is not None
        alpha = 1.0 / float(index)
        overlay_accumulator = overlay_accumulator * (1.0 - alpha) + image_bgr.astype(np.float32) * alpha

    assert overlay_accumulator is not None
    overlay_path = run_dir / "sample_all_raw.png"
    overlay_image = np.clip(np.round(overlay_accumulator), 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(overlay_path), overlay_image)
    return overlay_path


class TerminalCommandReader:
    def __init__(self) -> None:
        self._stop_event = Event()
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None or not sys.stdin or not sys.stdin.isatty():
            return
        self._thread = Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                break
            command = line.strip().lower()
            if not command:
                continue
            self._queue.put(command)

    def poll_all(self) -> list[str]:
        commands: list[str] = []
        while True:
            try:
                commands.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return commands

    def stop(self) -> None:
        self._stop_event.set()


class LiveRobotSnapshotReader:
    def __init__(self, robot: PiperDualCalibrationRobot, camera_key: str):
        self.robot = robot
        self.camera_key = camera_key
        self._stop_event = Event()
        self._ready_event = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._snapshot: dict[str, Any] | None = None
        self._frame_count = 0
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                controller_data, sensor_data = self.robot.get()
                image_rgb = np.asarray(sensor_data[self.camera_key]["color"], dtype=np.uint8).copy()
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                self._ready_event.set()
                return

            snapshot = {
                "timestamp": time.time(),
                "controller_data": controller_data,
                "image_rgb": image_rgb,
            }
            with self._lock:
                self._snapshot = snapshot
                self._frame_count += 1
            self._ready_event.set()

    def wait_until_ready(self, timeout_sec: float | None = None) -> bool:
        return self._ready_event.wait(timeout=timeout_sec)

    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def get_latest(self) -> tuple[dict[str, Any] | None, BaseException | None]:
        with self._lock:
            if self._snapshot is None:
                return None, self._error
            snapshot = {
                "timestamp": self._snapshot["timestamp"],
                "controller_data": self._snapshot["controller_data"],
                "image_rgb": self._snapshot["image_rgb"].copy(),
            }
            return snapshot, self._error

    def stop(self) -> None:
        self._stop_event.set()


def matrix_section(title: str, matrix: np.ndarray, decimals: int) -> str:
    return f"{title}\n{format_matrix(matrix, decimals=decimals)}"


def controller_name_for_arm(arm_name: str) -> str:
    if arm_name == "left_arm":
        return "slave_left_arm"
    if arm_name == "right_arm":
        return "slave_right_arm"
    raise ValueError(f"unsupported arm name: {arm_name}")


def other_arm_name(arm_name: str) -> str:
    if arm_name == "left_arm":
        return "right_arm"
    if arm_name == "right_arm":
        return "left_arm"
    raise ValueError(f"unsupported arm name: {arm_name}")


def pose_delta_metrics(reference_pose: np.ndarray, current_pose: np.ndarray) -> tuple[float, float]:
    reference_transform = pose6d_to_matrix(reference_pose)
    current_transform = pose6d_to_matrix(current_pose)
    translation_mm = float(np.linalg.norm(current_pose[:3] - reference_pose[:3]) * 1000.0)
    rotation_delta = reference_transform[:3, :3].T @ current_transform[:3, :3]
    rotation_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation_delta)[0])))
    return translation_mm, rotation_deg


def serialize_sample(sample: dict[str, Any], decimals: int) -> dict[str, Any]:
    return {
        "sample_index": int(sample["sample_index"]),
        "timestamp": sample["timestamp"],
        "arm_name": sample["arm_name"],
        "camera_name": sample["camera_name"],
        "ee_pose": [float(value) for value in sample["ee_pose"]],
        "arm_ee_pose_snapshot": {
            arm_key: [float(value) for value in arm_pose]
            for arm_key, arm_pose in sample["arm_ee_pose_snapshot"].items()
        },
        "reprojection_error_px": float(sample["reprojection_error_px"]),
        "marker_id": int(sample["marker_id"]),
        "image_raw_path": sample["image_raw_path"],
        "image_debug_path": sample["image_debug_path"],
        "image_preview_path": sample["image_preview_path"],
        "T_base_gripper": matrix_to_rounded_nested_list(sample["T_base_gripper"], decimals=decimals),
        "T_camera_tag": matrix_to_rounded_nested_list(sample["T_camera_tag"], decimals=decimals),
        "tag_corners_px": np.round(np.asarray(sample["tag_corners_px"], dtype=np.float64), decimals=3).tolist(),
    }


def save_samples_json(run_dir: Path, samples: list[dict[str, Any]], decimals: int) -> None:
    save_json(
        run_dir / "samples.json",
        {"samples": [serialize_sample(sample, decimals=decimals) for sample in samples]},
    )


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

    lines = [
        f"run_name: {run_name}",
        f"arm_name: {arm_name}",
        f"camera_name: cam_{camera_name}",
        f"sample_count: {len(samples)}",
        f"selected_method: {best_method}",
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
        f"- camera_translation_consistency_rmse_mm: {best_result['camera_translation_consistency_rmse_mm']:.4f}",
        f"- camera_translation_consistency_max_mm: {best_result['camera_translation_consistency_max_mm']:.4f}",
        f"- camera_rotation_consistency_rmse_deg: {best_result['camera_rotation_consistency_rmse_deg']:.4f}",
        f"- camera_rotation_consistency_max_deg: {best_result['camera_rotation_consistency_max_deg']:.4f}",
        f"- tag_in_gripper_translation_rmse_mm: {best_result['tag_in_gripper_translation_rmse_mm']:.4f}",
        f"- tag_in_gripper_translation_max_mm: {best_result['tag_in_gripper_translation_max_mm']:.4f}",
        f"- tag_in_gripper_rotation_rmse_deg: {best_result['tag_in_gripper_rotation_rmse_deg']:.4f}",
        f"- tag_in_gripper_rotation_max_deg: {best_result['tag_in_gripper_rotation_max_deg']:.4f}",
        f"- arm_translation_span_xyz_mm: {[round(value, 4) for value in best_result['motion_summary']['translation_span_xyz_mm']]}",
        f"- arm_translation_span_norm_mm: {best_result['motion_summary']['translation_span_norm_mm']:.4f}",
        f"- arm_max_relative_rotation_deg: {best_result['motion_summary']['max_relative_rotation_deg']:.4f}",
        "",
        matrix_section("T_base_camera (camera pose in arm base frame):", best_result["T_base_camera"], decimals),
        "",
        matrix_section("T_camera_base (arm base pose in camera frame):", best_result["T_camera_base"], decimals),
        "",
        matrix_section("T_gripper_tag (tag pose in end-effector frame):", best_result["T_gripper_tag"], decimals),
        "",
        matrix_section("T_tag_gripper (end-effector pose in tag frame):", best_result["T_tag_gripper"], decimals),
        "",
        "sample_T_base_gripper_list:",
        format_matrix_list([sample["T_base_gripper"] for sample in samples], decimals=decimals),
        "",
        "sample_T_camera_tag_list:",
        format_matrix_list([sample["T_camera_tag"] for sample in samples], decimals=decimals),
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
                f"cam_trans_rmse_mm={method_result['camera_translation_consistency_rmse_mm']:.4f}, "
                f"cam_rot_rmse_deg={method_result['camera_rotation_consistency_rmse_deg']:.4f}, "
                f"tag_trans_rmse_mm={method_result['tag_in_gripper_translation_rmse_mm']:.4f}, "
                f"tag_rot_rmse_deg={method_result['tag_in_gripper_rotation_rmse_deg']:.4f}",
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
        "arm_name": arm_name,
        "camera_name": f"cam_{camera_name}",
        "sample_count": len(samples),
        "selected_method": best_method,
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
            key: value
            for key, value in best_result.items()
            if key
            not in {
                "T_base_camera",
                "T_camera_base",
                "T_gripper_tag",
                "T_tag_gripper",
                "T_base_camera_per_sample",
                "success",
                "selection_score",
            }
        },
        "T_base_camera": matrix_to_rounded_nested_list(best_result["T_base_camera"], decimals=decimals),
        "T_camera_base": matrix_to_rounded_nested_list(best_result["T_camera_base"], decimals=decimals),
        "T_gripper_tag": matrix_to_rounded_nested_list(best_result["T_gripper_tag"], decimals=decimals),
        "T_tag_gripper": matrix_to_rounded_nested_list(best_result["T_tag_gripper"], decimals=decimals),
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
            "camera_translation_consistency_rmse_mm": method_result["camera_translation_consistency_rmse_mm"],
            "camera_translation_consistency_max_mm": method_result["camera_translation_consistency_max_mm"],
            "camera_rotation_consistency_rmse_deg": method_result["camera_rotation_consistency_rmse_deg"],
            "camera_rotation_consistency_max_deg": method_result["camera_rotation_consistency_max_deg"],
            "tag_in_gripper_translation_rmse_mm": method_result["tag_in_gripper_translation_rmse_mm"],
            "tag_in_gripper_translation_max_mm": method_result["tag_in_gripper_translation_max_mm"],
            "tag_in_gripper_rotation_rmse_deg": method_result["tag_in_gripper_rotation_rmse_deg"],
            "tag_in_gripper_rotation_max_deg": method_result["tag_in_gripper_rotation_max_deg"],
        }
    save_json(run_dir / "calibration_result.json", json_payload)


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
        f"arm={arm_name}  camera=cam_{camera_name}  samples={sample_count}",
        f"tag={'FOUND' if detection_found else 'NOT FOUND'}  dict={DEFAULT_TAG_DICTIONARY}  id={DEFAULT_TAG_ID}",
        f"ee_pose[m,rad]=[{', '.join(f'{value:.4f}' for value in ee_pose)}]",
        selected_delta_text,
        other_delta_text,
        "terminal only: c<Enter> save | q<Enter> quit",
        last_status,
    ]


def save_sample(
    run_dir: Path,
    sample_index: int,
    arm_name: str,
    camera_name: str,
    ee_pose: np.ndarray,
    arm_ee_pose_snapshot: dict[str, np.ndarray],
    detection: Any,
    image_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    preview_bgr_fullres: np.ndarray,
) -> dict[str, Any]:
    base_to_gripper = pose6d_to_matrix(ee_pose)
    raw_path = run_dir / f"sample_{sample_index:03d}_raw.png"
    debug_path = run_dir / f"sample_{sample_index:03d}_debug.png"
    preview_path = run_dir / f"sample_{sample_index:03d}_preview.png"
    cv2.imwrite(str(raw_path), image_bgr)
    cv2.imwrite(str(debug_path), annotated_bgr)
    cv2.imwrite(str(preview_path), preview_bgr_fullres)

    return {
        "sample_index": sample_index,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "arm_name": arm_name,
        "camera_name": f"cam_{camera_name}",
        "ee_pose": ee_pose.tolist(),
        "arm_ee_pose_snapshot": {
            arm_key: np.asarray(arm_pose, dtype=np.float64).reshape(6).tolist()
            for arm_key, arm_pose in arm_ee_pose_snapshot.items()
        },
        "reprojection_error_px": float(detection.reprojection_error_px),
        "marker_id": int(detection.marker_id),
        "image_raw_path": str(raw_path),
        "image_debug_path": str(debug_path),
        "image_preview_path": str(preview_path),
        "T_base_gripper": base_to_gripper,
        "T_camera_tag": detection.transform_camera_to_tag,
        "tag_corners_px": detection.corners_px,
    }


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

    calibration_result = run_hand_eye_calibration(samples)
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
    print("")
    print(f"selected_method: {calibration_result['best_method']}")
    print("T_base_camera:")
    print(format_matrix(best_result["T_base_camera"], decimals=decimals))
    print("")
    print("T_camera_base:")
    print(format_matrix(best_result["T_camera_base"], decimals=decimals))
    print("")
    print(
        "metrics: "
        f"reproj_rmse_px={best_result['reprojection_error_rmse_px']:.4f}, "
        f"cam_trans_rmse_mm={best_result['camera_translation_consistency_rmse_mm']:.4f}, "
        f"cam_rot_rmse_deg={best_result['camera_rotation_consistency_rmse_deg']:.4f}, "
        f"tag_trans_rmse_mm={best_result['tag_in_gripper_translation_rmse_mm']:.4f}, "
        f"tag_rot_rmse_deg={best_result['tag_in_gripper_rotation_rmse_deg']:.4f}"
    )
    print(f"report_path: {run_dir / 'calibration_report.txt'}")
    return 0


def run_offline_calibration(args: argparse.Namespace) -> int:
    run_dir = resolve_saved_run_dir(args.offline_run_dir)
    run_name, arm_name, camera_name, intrinsics, samples = load_saved_run(run_dir)

    print(f"offline_run_dir: {run_dir}")
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
    if args.offline_run_dir:
        return run_offline_calibration(args)

    arm_name = prompt_for_arm()
    run_name = f"{timestamp_for_path(datetime.now())}_{arm_name}"

    calib_root = ensure_dir(PROJECT_ROOT / "calib")
    runs_root = ensure_dir(calib_root / "runs")
    run_dir = ensure_dir(runs_root / run_name)

    condition = {
        "left_cam_serial": args.left_cam_serial,
        "high_cam_serial": args.high_cam_serial,
        "right_cam_serial": args.right_cam_serial,
    }

    robot = PiperDualCalibrationRobot(condition=condition, arm_name=arm_name, camera_name=args.camera)
    command_reader = TerminalCommandReader()
    snapshot_reader: LiveRobotSnapshotReader | None = None
    samples: list[dict[str, Any]] = []
    intrinsics: dict[str, Any] | None = None
    reference_arm_poses: dict[str, np.ndarray] | None = None
    last_status = "终端输入 c 回车采样，输入 q 回车结束并计算。"

    print(f"run_dir: {run_dir}")
    print(f"camera: cam_{args.camera}")
    print(
        "default_tag: "
        f"{DEFAULT_TAG_DICTIONARY}, id={DEFAULT_TAG_ID}, size={DEFAULT_TAG_SIZE_MM:.1f}mm"
    )
    print(
        "mapping: "
        f"l -> left_arm -> {controller_name_for_arm('left_arm')} -> {arm_to_can_bus('left_arm')}, "
        f"r -> right_arm -> {controller_name_for_arm('right_arm')} -> {arm_to_can_bus('right_arm')}"
    )
    print(
        "selected: "
        f"{arm_name} -> {controller_name_for_arm(arm_name)} -> {arm_to_can_bus(arm_name)}"
    )
    print("交互方式: 只使用终端。输入 c 后回车保存，输入 q 后回车结束。")
    print("窗口只用于预览，不处理键盘事件。")

    try:
        robot.set_up()
        command_reader.start()
        camera_key = f"cam_{args.camera}"
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
                camera_name=args.camera,
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
            save_requested = any(
                command in {"c", "save", "s"} for command in terminal_commands
            )
            quit_requested = any(
                command in {"q", "quit", "exit"} for command in terminal_commands
            )

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
                        camera_name=args.camera,
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
                    camera_name=args.camera,
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
            camera_name=args.camera,
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
