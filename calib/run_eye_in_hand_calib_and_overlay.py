#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import posixpath
import shutil
import subprocess
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

from calib.utils import ensure_dir, matrix_to_rounded_nested_list, pose6d_to_matrix, save_json
from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor


CALIB_DIR = PROJECT_ROOT / "calib"
CALIB_SCRIPT = CALIB_DIR / "dual_piper_arm_eye_in_hand_calib.py"
HOST_RUNS_ROOT = CALIB_DIR / "runs"

DEFAULT_HOST_WORKSPACE_ROOT = Path("/home/edemlab/challenge_ws/embodichain_ws")
DEFAULT_CONTAINER_WORKSPACE_ROOT = "/root/workspace"
DEFAULT_HOST_CONTAINER_RUNS_DIR = DEFAULT_HOST_WORKSPACE_ROOT / "calib_real_vs_sim" / "runs"
DEFAULT_HOST_REAL_MATERIAL_DIR = DEFAULT_HOST_WORKSPACE_ROOT / "calib_real_vs_sim" / "real_material"
DEFAULT_CONTAINER_REAL_MATERIAL_DIR = "/root/workspace/calib_real_vs_sim/real_material"
DEFAULT_REAL_CAM_LEFT_WRIST = DEFAULT_HOST_REAL_MATERIAL_DIR / "real_cam_left_wrist.jpg"
DEFAULT_REAL_CAM_RIGHT_WRIST = DEFAULT_HOST_REAL_MATERIAL_DIR / "real_cam_right_wrist.jpg"
DEFAULT_ARM_TXT = DEFAULT_HOST_REAL_MATERIAL_DIR / "arm_data.txt"
DEFAULT_ARM_NPY = DEFAULT_HOST_REAL_MATERIAL_DIR / "arm_data.npy"
DEFAULT_GT_CAPTURE_METADATA = DEFAULT_HOST_REAL_MATERIAL_DIR / "wrist_gt_capture_from_arm_data.json"

DEFAULT_CONTAINER_NAME = "embodichain"
DEFAULT_CONTAINER_PYTHON = "/root/miniconda3/envs/py310/bin/python"
DEFAULT_CONTAINER_RENDER_SCRIPT = "/root/workspace/calib_real_vs_sim/render_robot_calib_overlay.py"
DEFAULT_OVERLAY_SUBDIR = "overlay_real_material_eye_in_hand"
DEFAULT_OVERLAY_INPUT_SUBDIR = "_overlay_input_eye_in_hand"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute dual Piper wrist-camera eye-in-hand calibration results for existing runs, "
            "capture GT wrist images plus the matching dual-arm pose, then render real-vs-sim overlays "
            "inside the embodichain container."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help=(
            "existing eye-in-hand run directories to overlay; accepts absolute paths, cwd-relative paths, "
            "or bare run directory names under calib/runs. If omitted, the script runs a fresh "
            "eye-in-hand calibration first."
        ),
    )
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME)
    parser.add_argument("--container-python", default=DEFAULT_CONTAINER_PYTHON)
    parser.add_argument("--container-render-script", default=DEFAULT_CONTAINER_RENDER_SCRIPT)
    parser.add_argument("--host-workspace-root", type=Path, default=DEFAULT_HOST_WORKSPACE_ROOT)
    parser.add_argument("--container-workspace-root", default=DEFAULT_CONTAINER_WORKSPACE_ROOT)
    parser.add_argument("--host-container-runs-dir", type=Path, default=DEFAULT_HOST_CONTAINER_RUNS_DIR)
    parser.add_argument("--host-real-material-dir", type=Path, default=DEFAULT_HOST_REAL_MATERIAL_DIR)
    parser.add_argument("--container-real-material-dir", default=DEFAULT_CONTAINER_REAL_MATERIAL_DIR)
    parser.add_argument("--real-cam-left-wrist-image", type=Path, default=DEFAULT_REAL_CAM_LEFT_WRIST)
    parser.add_argument("--real-cam-right-wrist-image", type=Path, default=DEFAULT_REAL_CAM_RIGHT_WRIST)
    parser.add_argument("--arm-txt-path", type=Path, default=DEFAULT_ARM_TXT)
    parser.add_argument("--arm-npy-path", type=Path, default=DEFAULT_ARM_NPY)
    parser.add_argument("--arm-record-index", type=int, default=0)
    parser.add_argument("--gt-capture-metadata-path", type=Path, default=DEFAULT_GT_CAPTURE_METADATA)
    parser.add_argument("--capture-gt-images", action="store_true")
    parser.add_argument("--capture-warmup-frames", type=int, default=15)
    parser.add_argument("--capture-settle-timeout-sec", type=float, default=8.0)
    parser.add_argument("--capture-joint-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--left_cam_serial", type=str, default="344322073012")
    parser.add_argument("--high_cam_serial", type=str, default="323422071854")
    parser.add_argument("--right_cam_serial", type=str, default="335522070790")
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-headless", action="store_true", help="do not pass --headless to renderer")
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    parser.add_argument("--overlay-subdir", default=DEFAULT_OVERLAY_SUBDIR)
    parser.add_argument("--overlay-input-subdir", default=DEFAULT_OVERLAY_INPUT_SUBDIR)
    parser.add_argument("--calib-camera-convention", default="ros", choices=["ros", "opengl", "world"])
    parser.add_argument("--joint6-limit-override", type=float, default=None)
    parser.add_argument("--max-reproj-rmse-px", type=float, default=2.0)
    parser.add_argument("--max-cam-in-gripper-trans-rmse-mm", type=float, default=50.0)
    parser.add_argument("--max-cam-in-gripper-rot-rmse-deg", type=float, default=5.0)
    parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="render whenever calibration_result.json exists, without metric thresholds",
    )
    return parser.parse_args()


def ensure_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")
    return path


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


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def direct_run_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def newest_run_dir(root: Path, *, created_after: float, previous_dirs: set[Path]) -> Path:
    current_dirs = direct_run_dirs(root)
    new_dirs = sorted(current_dirs - previous_dirs, key=lambda path: path.stat().st_mtime, reverse=True)
    if new_dirs:
        return new_dirs[0]

    recent_dirs = sorted(
        (path for path in current_dirs if path.stat().st_mtime >= created_after - 1.0),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if recent_dirs:
        return recent_dirs[0]

    all_dirs = sorted(current_dirs, key=lambda path: path.stat().st_mtime, reverse=True)
    if all_dirs:
        return all_dirs[0]
    raise FileNotFoundError(f"no run directories found under {root}")


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def map_host_path_to_container(host_path: Path, host_workspace_root: Path, container_workspace_root: str) -> str:
    host_resolved = host_path.expanduser().resolve()
    workspace_root = host_workspace_root.expanduser().resolve()
    if not is_relative_to(host_resolved, workspace_root):
        raise ValueError(f"path {host_resolved} is not under mounted workspace {workspace_root}")
    relative_path = host_resolved.relative_to(workspace_root)
    return posixpath.join(container_workspace_root, relative_path.as_posix())


def camera_name_for_arm(arm_name: str) -> str:
    if arm_name == "left_arm":
        return "cam_left_wrist"
    if arm_name == "right_arm":
        return "cam_right_wrist"
    raise ValueError(f"unsupported arm name: {arm_name}")


def host_real_image_for_arm(args: argparse.Namespace, arm_name: str) -> Path:
    if arm_name == "left_arm":
        return args.real_cam_left_wrist_image.expanduser().resolve()
    if arm_name == "right_arm":
        return args.real_cam_right_wrist_image.expanduser().resolve()
    raise ValueError(f"unsupported arm name: {arm_name}")


def format_pose_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in np.asarray(values, dtype=np.float64).reshape(-1)) + "]"


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


def load_arm_pose_payload(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"unexpected arm_data payload in {path}")
    return payload


def select_arm_pose_record(payload: dict[str, Any], record_index: int) -> dict[str, Any]:
    records = payload["records"]
    if record_index < 0 or record_index >= len(records):
        raise IndexError(f"arm_record_index={record_index} is out of range for {len(records)} records")
    return records[record_index]


def gripper_command_from_scalar(raw_value: float) -> float:
    value = float(raw_value)
    if value <= 0.05:
        return float(np.clip(value / 0.05 if value > 0.0 else 0.0, 0.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def ee_pose_key_for_arm(arm_name: str) -> str:
    if arm_name == "left_arm":
        return "left_ee_pose_xyz_rpy"
    if arm_name == "right_arm":
        return "right_ee_pose_xyz_rpy"
    raise ValueError(f"unsupported arm name: {arm_name}")


def wait_for_joint_targets(
    left_arm: PiperController,
    right_arm: PiperController,
    left_target: np.ndarray,
    right_target: np.ndarray,
    timeout_sec: float,
    tolerance_rad: float,
) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        left_error = np.max(np.abs(np.asarray(left_arm.get_state()["joint"], dtype=np.float64) - left_target))
        right_error = np.max(np.abs(np.asarray(right_arm.get_state()["joint"], dtype=np.float64) - right_target))
        if left_error <= tolerance_rad and right_error <= tolerance_rad:
            return
        time.sleep(0.1)
    print(
        "capture_warning: timed out waiting for arm targets; "
        f"left_max_joint_error={left_error:.4f}rad right_max_joint_error={right_error:.4f}rad"
    )


def capture_gt_material(args: argparse.Namespace) -> dict[str, Any]:
    left_arm = PiperController("slave_left_arm")
    right_arm = PiperController("slave_right_arm")
    left_cam = RealsenseSensor("cam_left")
    right_cam = RealsenseSensor("cam_right")

    host_real_material_dir = ensure_dir(args.host_real_material_dir.expanduser().resolve())
    left_image_path = args.real_cam_left_wrist_image.expanduser().resolve()
    right_image_path = args.real_cam_right_wrist_image.expanduser().resolve()
    arm_txt_path = ensure_file(args.arm_txt_path.expanduser().resolve())
    arm_npy_path = ensure_file(args.arm_npy_path.expanduser().resolve())
    metadata_path = args.gt_capture_metadata_path.expanduser().resolve()
    left_image_path.parent.mkdir(parents=True, exist_ok=True)
    right_image_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    arm_payload = load_arm_pose_payload(arm_npy_path)
    selected_record = select_arm_pose_record(arm_payload, args.arm_record_index)
    left_joint_target = np.asarray(selected_record["left_arm_joints"], dtype=np.float64).reshape(6)
    right_joint_target = np.asarray(selected_record["right_arm_joints"], dtype=np.float64).reshape(6)
    left_gripper_target = gripper_command_from_scalar(float(selected_record["left_gripper_scalar"]))
    right_gripper_target = gripper_command_from_scalar(float(selected_record["right_gripper_scalar"]))

    try:
        left_arm.set_up("can0")
        right_arm.set_up("can1")

        left_cam.set_up(args.left_cam_serial)
        right_cam.set_up(args.right_cam_serial)
        left_cam.set_collect_info(["color"])
        right_cam.set_collect_info(["color"])

        latest_left_rgb: np.ndarray | None = None
        latest_right_rgb: np.ndarray | None = None
        for _ in range(max(args.capture_warmup_frames, 1)):
            latest_left_rgb = np.asarray(left_cam.get_image()["color"], dtype=np.uint8)
            latest_right_rgb = np.asarray(right_cam.get_image()["color"], dtype=np.uint8)

        left_arm.set_gripper(left_gripper_target)
        right_arm.set_gripper(right_gripper_target)
        left_arm.set_joint(left_joint_target)
        right_arm.set_joint(right_joint_target)
        wait_for_joint_targets(
            left_arm=left_arm,
            right_arm=right_arm,
            left_target=left_joint_target,
            right_target=right_joint_target,
            timeout_sec=args.capture_settle_timeout_sec,
            tolerance_rad=args.capture_joint_tolerance_rad,
        )

        time.sleep(0.5)
        for _ in range(max(args.capture_warmup_frames // 2, 3)):
            latest_left_rgb = np.asarray(left_cam.get_image()["color"], dtype=np.uint8)
            latest_right_rgb = np.asarray(right_cam.get_image()["color"], dtype=np.uint8)

        if latest_left_rgb is None or latest_right_rgb is None:
            raise RuntimeError("failed to capture GT wrist images")

        controller_data = {
            "left_arm": left_arm.get_state(),
            "right_arm": right_arm.get_state(),
        }

        cv2.imwrite(str(left_image_path), cv2.cvtColor(latest_left_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(right_image_path), cv2.cvtColor(latest_right_rgb, cv2.COLOR_RGB2BGR))

        metadata = {
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "host_real_material_dir": str(host_real_material_dir),
            "left_cam_serial": args.left_cam_serial,
            "right_cam_serial": args.right_cam_serial,
            "left_image_path": str(left_image_path),
            "right_image_path": str(right_image_path),
            "arm_txt_path": str(arm_txt_path),
            "arm_npy_path": str(arm_npy_path),
            "arm_record_index": int(args.arm_record_index),
            "selected_record": to_jsonable(selected_record),
            "target_commands": {
                "left_arm_joints": left_joint_target.tolist(),
                "right_arm_joints": right_joint_target.tolist(),
                "left_gripper_opening": float(left_gripper_target),
                "right_gripper_opening": float(right_gripper_target),
            },
            "actual_controller_data": to_jsonable(controller_data),
        }
        save_json(metadata_path, metadata)
        print(f"captured_gt_metadata: {metadata_path}")
        print(f"captured_left_wrist_image: {left_image_path}")
        print(f"captured_right_wrist_image: {right_image_path}")
        print(f"capture_arm_record_index: {args.arm_record_index}")
        print(f"capture_left_target_ee_pose: {format_pose_vector(np.asarray(selected_record['left_ee_pose_xyz_rpy'], dtype=np.float64))}")
        print(
            f"capture_right_target_ee_pose: "
            f"{format_pose_vector(np.asarray(selected_record['right_ee_pose_xyz_rpy'], dtype=np.float64))}"
        )
        return metadata
    finally:
        left_cam.cleanup()
        right_cam.cleanup()


def load_gt_capture_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = ensure_file(args.gt_capture_metadata_path.expanduser().resolve())
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ensure_file(Path(metadata["left_image_path"]))
    ensure_file(Path(metadata["right_image_path"]))
    ensure_file(Path(metadata.get("arm_txt_path", args.arm_txt_path.expanduser().resolve())))
    arm_npy_path = ensure_file(Path(metadata.get("arm_npy_path", args.arm_npy_path.expanduser().resolve())))
    if "arm_record_index" not in metadata:
        metadata["arm_record_index"] = int(args.arm_record_index)
    if "selected_record" not in metadata:
        arm_payload = load_arm_pose_payload(arm_npy_path)
        metadata["selected_record"] = to_jsonable(select_arm_pose_record(arm_payload, int(metadata["arm_record_index"])))
    return metadata


def run_live_calibration(args: argparse.Namespace) -> Path:
    HOST_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    before_dirs = direct_run_dirs(HOST_RUNS_ROOT)
    start_time = time.time()
    command = [
        sys.executable,
        str(CALIB_SCRIPT),
        "--left_cam_serial",
        args.left_cam_serial,
        "--high_cam_serial",
        args.high_cam_serial,
        "--right_cam_serial",
        args.right_cam_serial,
        "--preview-scale",
        str(args.preview_scale),
        "--warmup-frames",
        str(args.warmup_frames),
        "--decimals",
        str(args.decimals),
    ]
    print("running_eye_in_hand_calibration:")
    print(" ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    run_dir = newest_run_dir(HOST_RUNS_ROOT, created_after=start_time, previous_dirs=before_dirs)
    print(f"calibration_run_dir: {run_dir}")
    return run_dir


def recompute_calibration_result(run_dir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(CALIB_SCRIPT),
        str(run_dir),
        "--decimals",
        str(args.decimals),
    ]
    print("recomputing_eye_in_hand_result:")
    print(" ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def load_calibration_result(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "calibration_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"calibration result not found: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def metric_value(result: dict[str, Any], key: str) -> float:
    value = result.get("best_metrics", {}).get(key)
    if value is None:
        raise KeyError(f"missing metric in calibration_result.json: best_metrics.{key}")
    return float(value)


def validate_calibration_result(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    result = load_calibration_result(run_dir)
    sample_count = int(result.get("sample_count", 0))
    selected_method = result.get("selected_method")
    if sample_count < 3:
        raise RuntimeError(f"calibration has too few samples: {sample_count}")
    if not selected_method:
        raise RuntimeError("calibration_result.json is missing selected_method")
    if args.no_quality_gate:
        print(f"quality_gate: skipped (selected_method={selected_method}, sample_count={sample_count})")
        return result

    reproj = metric_value(result, "reprojection_error_rmse_px")
    cam_trans = metric_value(result, "camera_in_gripper_translation_rmse_mm")
    cam_rot = metric_value(result, "camera_in_gripper_rotation_rmse_deg")
    failures = []
    if reproj > args.max_reproj_rmse_px:
        failures.append(f"reproj_rmse_px={reproj:.4f} > {args.max_reproj_rmse_px:.4f}")
    if cam_trans > args.max_cam_in_gripper_trans_rmse_mm:
        failures.append(
            f"cam_in_gripper_trans_rmse_mm={cam_trans:.4f} > {args.max_cam_in_gripper_trans_rmse_mm:.4f}"
        )
    if cam_rot > args.max_cam_in_gripper_rot_rmse_deg:
        failures.append(
            f"cam_in_gripper_rot_rmse_deg={cam_rot:.4f} > {args.max_cam_in_gripper_rot_rmse_deg:.4f}"
        )
    if failures:
        raise RuntimeError("calibration quality gate failed: " + "; ".join(failures))
    print(
        "quality_gate: passed "
        f"(method={selected_method}, samples={sample_count}, "
        f"reproj={reproj:.4f}px, cam_in_gripper_trans={cam_trans:.4f}mm, cam_in_gripper_rot={cam_rot:.4f}deg)"
    )
    return result


def copy_run_to_container_visible_dir(run_dir: Path, args: argparse.Namespace) -> tuple[Path, bool]:
    host_container_runs_dir = ensure_dir(args.host_container_runs_dir.expanduser().resolve())
    run_dir_resolved = run_dir.resolve()
    if is_relative_to(run_dir_resolved, host_container_runs_dir):
        print(f"container_access: run already under mounted workspace: {run_dir_resolved}")
        return run_dir_resolved, False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    mirror_run_dir = host_container_runs_dir / f"{run_dir.name}__overlay_tmp_{timestamp}"
    shutil.copytree(run_dir_resolved, mirror_run_dir, symlinks=False)
    print(f"container_access: mirrored {run_dir_resolved} -> {mirror_run_dir}")
    return mirror_run_dir, True


def build_overlay_input_dir(
    source_run_dir: Path,
    target_dir: Path,
    overlay_ee_pose: np.ndarray,
    metadata_path: Path,
    arm_npy_path: Path,
    arm_record_index: int,
    decimals: int,
) -> None:
    source_result = load_calibration_result(source_run_dir)
    supplemental = source_result.get("supplemental_eye_in_hand") or {}
    T_gripper_camera = supplemental.get("T_gripper_camera")
    if T_gripper_camera is None:
        raise KeyError(
            f"calibration_result.json in {source_run_dir} is missing supplemental_eye_in_hand.T_gripper_camera"
        )

    T_base_gripper = pose6d_to_matrix(overlay_ee_pose)
    T_base_camera = T_base_gripper @ np.asarray(T_gripper_camera, dtype=np.float64)
    T_camera_base = np.linalg.inv(T_base_camera)

    payload = copy.deepcopy(source_result)
    payload["reference_sample_index"] = -1
    payload["reference_ee_pose"] = [float(value) for value in np.asarray(overlay_ee_pose, dtype=np.float64).reshape(6)]
    payload["T_base_camera"] = matrix_to_rounded_nested_list(T_base_camera, decimals=decimals)
    payload["T_camera_base"] = matrix_to_rounded_nested_list(T_camera_base, decimals=decimals)
    payload["overlay_pose_source"] = {
        "type": "arm_data_record",
        "arm_npy_path": str(arm_npy_path),
        "arm_record_index": int(arm_record_index),
        "metadata_path": str(metadata_path),
    }

    ensure_dir(target_dir)
    shutil.copy2(source_run_dir / "camera_intrinsics.json", target_dir / "camera_intrinsics.json")
    save_json(target_dir / "calibration_result.json", payload)


def sync_overlay_outputs_back(source_output_dir: Path, target_output_dir: Path) -> None:
    shutil.copytree(source_output_dir, target_output_dir, dirs_exist_ok=True)
    print(f"synced_overlay_outputs: {source_output_dir} -> {target_output_dir}")


def run_overlay_for_run(run_dir: Path, args: argparse.Namespace, gt_metadata: dict[str, Any]) -> None:
    recompute_calibration_result(run_dir, args)
    result = validate_calibration_result(run_dir, args)
    arm_name = str(result.get("arm_name"))
    sensor_name = camera_name_for_arm(arm_name)
    mirror_run_dir, needs_copy_back = copy_run_to_container_visible_dir(run_dir, args)

    overlay_input_dir = mirror_run_dir / args.overlay_input_subdir
    output_dir = mirror_run_dir / args.overlay_subdir
    selected_record = gt_metadata["selected_record"]
    overlay_ee_pose = np.asarray(selected_record[ee_pose_key_for_arm(arm_name)], dtype=np.float64).reshape(6)
    metadata_path = args.gt_capture_metadata_path.expanduser().resolve()
    build_overlay_input_dir(
        source_run_dir=mirror_run_dir,
        target_dir=overlay_input_dir,
        overlay_ee_pose=overlay_ee_pose,
        metadata_path=metadata_path,
        arm_npy_path=args.arm_npy_path.expanduser().resolve(),
        arm_record_index=int(gt_metadata["arm_record_index"]),
        decimals=args.decimals,
    )

    container_overlay_input_dir = map_host_path_to_container(
        overlay_input_dir, args.host_workspace_root, args.container_workspace_root
    )
    container_output_dir = map_host_path_to_container(
        output_dir, args.host_workspace_root, args.container_workspace_root
    )
    container_real_image_path = map_host_path_to_container(
        host_real_image_for_arm(args, arm_name), args.host_workspace_root, args.container_workspace_root
    )
    container_gt_arm_txt_path = map_host_path_to_container(
        args.arm_txt_path.expanduser().resolve(), args.host_workspace_root, args.container_workspace_root
    )
    container_gt_arm_npy_path = map_host_path_to_container(
        args.arm_npy_path.expanduser().resolve(), args.host_workspace_root, args.container_workspace_root
    )

    sensor_dir_arg = f"--{sensor_name}_dir"
    real_image_arg = f"--real_{sensor_name}_image"
    command = [
        "docker",
        "exec",
        args.container_name,
        args.container_python,
        args.container_render_script,
        "--device",
        args.device,
        "--calib_camera_convention",
        args.calib_camera_convention,
        "--arm_txt_path",
        container_gt_arm_txt_path,
        "--arm_npy_path",
        container_gt_arm_npy_path,
        "--arm_record_index",
        str(int(gt_metadata["arm_record_index"])),
        "--overlay_alpha",
        str(args.overlay_alpha),
        sensor_dir_arg,
        container_overlay_input_dir,
        real_image_arg,
        container_real_image_path,
        "--output_dir",
        container_output_dir,
    ]
    if not args.no_headless:
        command.append("--headless")
    if args.joint6_limit_override is not None:
        command.extend(["--joint6_limit_override", str(args.joint6_limit_override)])

    print("running_eye_in_hand_overlay:")
    print(" ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    target_output_dir = run_dir / args.overlay_subdir
    sync_overlay_outputs_back(output_dir, target_output_dir)
    if needs_copy_back:
        try:
            remove_path(mirror_run_dir)
        except PermissionError as exc:
            print(f"cleanup_warning: could not remove temporary mirror directory {mirror_run_dir}: {exc}")
    print(f"done: overlay outputs are available under {target_output_dir}")


def main() -> int:
    args = parse_args()

    run_dirs = [resolve_saved_run_dir(run_dir_arg) for run_dir_arg in args.run_dirs]
    if not run_dirs:
        run_dirs = [run_live_calibration(args)]

    if args.capture_gt_images:
        gt_metadata = capture_gt_material(args)
    else:
        gt_metadata = load_gt_capture_metadata(args)

    for run_dir in run_dirs:
        print(f"overlay_run_dir: {run_dir}")
        run_overlay_for_run(run_dir, args, gt_metadata)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
