#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import posixpath
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calib.dual_piper_arm_extrinsic_calib import load_saved_run, resolve_saved_run_dir
from calib.utils import (
    ensure_dir,
    invert_transform,
    matrix_to_rounded_nested_list,
    pose6d_to_matrix,
    run_eye_in_hand_calibration,
    save_json,
)


DEFAULT_HOST_WORKSPACE_ROOT = Path("/home/edemlab/challenge_ws/embodichain_ws")
DEFAULT_CONTAINER_WORKSPACE_ROOT = "/root/workspace"
DEFAULT_HOST_CONTAINER_RUNS_DIR = DEFAULT_HOST_WORKSPACE_ROOT / "calib_real_vs_sim" / "runs"
DEFAULT_HOST_REAL_MATERIAL_DIR = DEFAULT_HOST_WORKSPACE_ROOT / "calib_real_vs_sim" / "real_material"
DEFAULT_ARM_TXT = DEFAULT_HOST_REAL_MATERIAL_DIR / "arm_data.txt"
DEFAULT_ARM_NPY = DEFAULT_HOST_REAL_MATERIAL_DIR / "arm_data.npy"
DEFAULT_REAL_CAM_LEFT_WRIST = DEFAULT_HOST_REAL_MATERIAL_DIR / "real_cam_left_wrist.jpg"
DEFAULT_REAL_CAM_RIGHT_WRIST = DEFAULT_HOST_REAL_MATERIAL_DIR / "real_cam_right_wrist.jpg"
DEFAULT_CONTAINER_NAME = "embodichain"
DEFAULT_CONTAINER_PYTHON = "/root/miniconda3/envs/py310/bin/python"
DEFAULT_CONTAINER_RENDER_SCRIPT = "/root/workspace/calib_real_vs_sim/render_robot_calib_overlay.py"
DEFAULT_OVERLAY_ALPHA = 0.5
DEFAULT_CONTACT_COLUMNS = 3


@dataclass
class VariantSpec:
    name: str
    description: str
    calib_camera_convention: str
    T_base_camera: np.ndarray
    local_pose_opengl: np.ndarray
    source_method: str | None
    source_kind: str
    world_z_offset_m: float = 0.0
    notes: dict[str, Any] | None = None
    T_gripper_camera_assumed: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run all OpenCV hand-eye methods for an existing wrist eye-in-hand run, "
            "render each result in simulation, and add diagnostic pose probes such as "
            "challenge default wrist mount, inverse transform, and world-z offsets."
        )
    )
    parser.add_argument("run_dir", help="existing eye-in-hand run directory")
    parser.add_argument("--arm-txt-path", type=Path, default=DEFAULT_ARM_TXT)
    parser.add_argument("--arm-npy-path", type=Path, default=DEFAULT_ARM_NPY)
    parser.add_argument("--arm-record-index", type=int, default=0)
    parser.add_argument("--host-workspace-root", type=Path, default=DEFAULT_HOST_WORKSPACE_ROOT)
    parser.add_argument("--container-workspace-root", default=DEFAULT_CONTAINER_WORKSPACE_ROOT)
    parser.add_argument("--host-container-runs-dir", type=Path, default=DEFAULT_HOST_CONTAINER_RUNS_DIR)
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME)
    parser.add_argument("--container-python", default=DEFAULT_CONTAINER_PYTHON)
    parser.add_argument("--container-render-script", default=DEFAULT_CONTAINER_RENDER_SCRIPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument(
        "--world-z-offsets-mm",
        type=float,
        nargs="*",
        default=[50.0, 100.0, -50.0],
        help="diagnostic world-z offsets applied on top of the selected method result",
    )
    parser.add_argument(
        "--export-subdir-prefix",
        default="diagnose_eye_in_hand_methods",
        help="prefix for the host-side export directory created under the source run dir",
    )
    return parser.parse_args()


def ensure_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")
    return path


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def map_host_path_to_container(host_path: Path, host_workspace_root: Path, container_workspace_root: str) -> str:
    host_resolved = host_path.expanduser().resolve()
    workspace_root = host_workspace_root.expanduser().resolve()
    if not is_relative_to(host_resolved, workspace_root):
        raise ValueError(f"path {host_resolved} is not under mounted workspace {workspace_root}")
    relative_path = host_resolved.relative_to(workspace_root)
    return posixpath.join(container_workspace_root, relative_path.as_posix())


def map_container_path_to_host(container_path: str | None, host_workspace_root: Path, container_workspace_root: str) -> str | None:
    if not container_path:
        return container_path
    normalized_root = container_workspace_root.rstrip("/")
    if not container_path.startswith(normalized_root + "/") and container_path != normalized_root:
        return container_path
    relative_path = container_path[len(normalized_root) :].lstrip("/")
    return str((host_workspace_root / relative_path).resolve())


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


def sanitize_name(text: str) -> str:
    safe = []
    for char in text:
        if char.isalnum() or char in {"_", "-", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


def camera_sensor_name_for_arm(arm_name: str) -> str:
    if arm_name == "left_arm":
        return "cam_left_wrist"
    if arm_name == "right_arm":
        return "cam_right_wrist"
    raise ValueError(f"unsupported arm name: {arm_name}")


def real_image_for_arm(arm_name: str) -> Path:
    if arm_name == "left_arm":
        return DEFAULT_REAL_CAM_LEFT_WRIST
    if arm_name == "right_arm":
        return DEFAULT_REAL_CAM_RIGHT_WRIST
    raise ValueError(f"unsupported arm name: {arm_name}")


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


def ee_pose_key_for_arm(arm_name: str) -> str:
    if arm_name == "left_arm":
        return "left_ee_pose_xyz_rpy"
    if arm_name == "right_arm":
        return "right_ee_pose_xyz_rpy"
    raise ValueError(f"unsupported arm name: {arm_name}")


def pose_opengl_to_ros(pose_opengl: np.ndarray) -> np.ndarray:
    pose_ros = np.asarray(pose_opengl, dtype=np.float64).copy()
    pose_ros[:3, 1] *= -1.0
    pose_ros[:3, 2] *= -1.0
    return pose_ros


def pose_ros_to_opengl(pose_ros: np.ndarray) -> np.ndarray:
    pose_gl = np.asarray(pose_ros, dtype=np.float64).copy()
    pose_gl[:3, 1] *= -1.0
    pose_gl[:3, 2] *= -1.0
    return pose_gl


def make_transform_from_wxyz(pos_xyz: list[float], quat_wxyz: list[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    w, x, y, z = [float(value) for value in quat_wxyz]
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
    return pose


def rotation_error_deg(reference: np.ndarray, query: np.ndarray) -> float:
    delta = reference.T @ query
    trace_value = float(np.trace(delta))
    cos_angle = max(-1.0, min(1.0, (trace_value - 1.0) * 0.5))
    return float(math.degrees(math.acos(cos_angle)))


def build_contact_sheet(
    title: str,
    image_paths: list[tuple[str, Path]],
    output_path: Path,
    columns: int = DEFAULT_CONTACT_COLUMNS,
    thumb_size: tuple[int, int] = (320, 240),
) -> Path | None:
    valid_items: list[tuple[str, np.ndarray]] = []
    for label, image_path in image_paths:
        if not image_path.is_file():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            rgb = image[:, :, :3].astype(np.float32)
            image = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)
        valid_items.append((label, image))
    if not valid_items:
        return None

    thumb_w, thumb_h = thumb_size
    header_h = 56
    label_h = 36
    margin = 12
    columns = max(columns, 1)
    rows = int(math.ceil(len(valid_items) / columns))
    canvas_w = columns * thumb_w + (columns + 1) * margin
    canvas_h = header_h + rows * (thumb_h + label_h + margin) + margin
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    cv2.putText(
        canvas,
        title,
        (margin, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    for index, (label, image) in enumerate(valid_items):
        row = index // columns
        col = index % columns
        x0 = margin + col * (thumb_w + margin)
        y0 = header_h + row * (thumb_h + label_h + margin)
        resized = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = resized
        cv2.rectangle(canvas, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (180, 180, 180), 1)
        cv2.putText(
            canvas,
            label,
            (x0 + 4, y0 + thumb_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def default_local_pose_opengl_for_sensor(sensor_name: str) -> np.ndarray:
    challenge_config_path = DEFAULT_HOST_WORKSPACE_ROOT / "Embodied_Challenge" / "configs" / "pour_water_dual" / "gym_config_dual.json"
    raw_text = challenge_config_path.read_text(encoding="utf-8")
    config = None
    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            config = yaml.safe_load(raw_text)
        except yaml.YAMLError:
            config = None
    if isinstance(config, dict):
        sensors = config.get("sensor", [])
        for sensor_cfg in sensors:
            if sensor_cfg.get("uid") != sensor_name:
                continue
            extrinsics = sensor_cfg.get("extrinsics", {})
            if "parent" not in extrinsics:
                raise ValueError(f"sensor {sensor_name} is not a parented wrist camera in {challenge_config_path}")
            return make_transform_from_wxyz(extrinsics["pos"], extrinsics["quat"])

    pattern = re.compile(
        rf'"uid"\s*:\s*"{re.escape(sensor_name)}".*?"extrinsics"\s*:\s*\{{.*?"parent"\s*:\s*"([^"]+)".*?"pos"\s*:\s*\[([^\]]+)\].*?"quat"\s*:\s*\[([^\]]+)\]',
        re.DOTALL,
    )
    match = pattern.search(raw_text)
    if not match:
        raise ValueError(f"sensor {sensor_name} not found in {challenge_config_path}")
    pos_xyz = [float(value.strip()) for value in match.group(2).split(",")]
    quat_wxyz = [float(value.strip()) for value in match.group(3).split(",")]
    return make_transform_from_wxyz(pos_xyz, quat_wxyz)


def make_minimal_calibration_payload(
    source_result: dict[str, Any],
    T_base_camera: np.ndarray,
    reference_ee_pose: np.ndarray,
    variant: VariantSpec,
    decimals: int,
) -> dict[str, Any]:
    payload = {
        "run_name": f"{source_result.get('run_name', 'run')}__{variant.name}",
        "calibration_mode": "eye_in_hand_diagnostic",
        "arm_name": source_result["arm_name"],
        "camera_name": source_result["camera_name"],
        "sample_count": int(source_result.get("sample_count", 0)),
        "selected_method": variant.source_method or source_result.get("selected_method"),
        "reference_sample_index": -1,
        "reference_ee_pose": [float(value) for value in reference_ee_pose.reshape(6)],
        "T_base_camera": matrix_to_rounded_nested_list(T_base_camera, decimals=decimals),
        "T_camera_base": matrix_to_rounded_nested_list(invert_transform(T_base_camera), decimals=decimals),
        "diagnostic_variant": {
            "name": variant.name,
            "description": variant.description,
            "source_kind": variant.source_kind,
            "source_method": variant.source_method,
            "world_z_offset_m": float(variant.world_z_offset_m),
            "calib_camera_convention": variant.calib_camera_convention,
            "notes": to_jsonable(variant.notes or {}),
        },
    }
    if variant.T_gripper_camera_assumed is not None:
        payload["supplemental_eye_in_hand"] = {
            "T_gripper_camera": matrix_to_rounded_nested_list(
                variant.T_gripper_camera_assumed, decimals=decimals
            ),
            "T_camera_gripper": matrix_to_rounded_nested_list(
                invert_transform(variant.T_gripper_camera_assumed), decimals=decimals
            ),
        }
    return payload


def build_variants(
    source_result: dict[str, Any],
    arm_name: str,
    selected_record: dict[str, Any],
    eye_in_hand_result: dict[str, Any],
    world_z_offsets_mm: list[float],
) -> list[VariantSpec]:
    reference_ee_pose = np.asarray(selected_record[ee_pose_key_for_arm(arm_name)], dtype=np.float64).reshape(6)
    T_base_gripper = pose6d_to_matrix(reference_ee_pose)
    sensor_name = camera_sensor_name_for_arm(arm_name)
    default_local_pose_gl = default_local_pose_opengl_for_sensor(sensor_name)
    default_local_pose_ros = pose_opengl_to_ros(default_local_pose_gl)

    variants: list[VariantSpec] = []

    default_base_camera = T_base_gripper @ default_local_pose_ros
    variants.append(
        VariantSpec(
            name="challenge_default_mount",
            description="Challenge default wrist camera mount from gym_config_dual.json",
            calib_camera_convention="ros",
            T_base_camera=default_base_camera,
            local_pose_opengl=default_local_pose_gl,
            source_method=None,
            source_kind="challenge_default_mount",
            notes={"parent_sensor_name": sensor_name},
            T_gripper_camera_assumed=default_local_pose_ros,
        )
    )

    all_results = eye_in_hand_result["all_results"]
    for method_name, method_result in all_results.items():
        if not method_result.get("success"):
            continue
        T_gripper_camera = np.asarray(method_result["T_gripper_camera"], dtype=np.float64)
        T_base_camera = T_base_gripper @ T_gripper_camera
        variants.append(
            VariantSpec(
                name=f"method_{method_name.lower()}",
                description=f"{method_name} raw eye-in-hand solution",
                calib_camera_convention="ros",
                T_base_camera=T_base_camera,
                local_pose_opengl=pose_ros_to_opengl(T_gripper_camera),
                source_method=method_name,
                source_kind="method_result",
                notes={
                    "tag_in_base_translation_rmse_mm": float(method_result["tag_in_base_translation_rmse_mm"]),
                    "tag_in_base_rotation_rmse_deg": float(method_result["tag_in_base_rotation_rmse_deg"]),
                    "camera_in_gripper_translation_rmse_mm": float(
                        method_result["camera_in_gripper_translation_rmse_mm"]
                    ),
                    "camera_in_gripper_rotation_rmse_deg": float(
                        method_result["camera_in_gripper_rotation_rmse_deg"]
                    ),
                },
                T_gripper_camera_assumed=T_gripper_camera,
            )
        )

    selected_method = str(eye_in_hand_result["best_method"])
    selected_result = eye_in_hand_result["best_result"]
    selected_T_gripper_camera = np.asarray(selected_result["T_gripper_camera"], dtype=np.float64)
    selected_T_base_camera = T_base_gripper @ selected_T_gripper_camera

    inverse_T_gripper_camera = invert_transform(selected_T_gripper_camera)
    variants.append(
        VariantSpec(
            name=f"{selected_method.lower()}_inverse_mount",
            description=f"Inverse of selected {selected_method} solution, used only as a directionality probe",
            calib_camera_convention="ros",
            T_base_camera=T_base_gripper @ inverse_T_gripper_camera,
            local_pose_opengl=pose_ros_to_opengl(inverse_T_gripper_camera),
            source_method=selected_method,
            source_kind="selected_inverse_probe",
            T_gripper_camera_assumed=inverse_T_gripper_camera,
        )
    )

    variants.append(
        VariantSpec(
            name=f"{selected_method.lower()}_as_opengl",
            description=f"Selected {selected_method} solution rendered without ROS->OpenGL axis conversion",
            calib_camera_convention="opengl",
            T_base_camera=selected_T_base_camera,
            local_pose_opengl=selected_T_gripper_camera,
            source_method=selected_method,
            source_kind="selected_convention_probe",
            notes={"probe": "treat_selected_transform_as_opengl"},
            T_gripper_camera_assumed=selected_T_gripper_camera,
        )
    )

    for offset_mm in world_z_offsets_mm:
        offset_m = float(offset_mm) / 1000.0
        shifted = selected_T_base_camera.copy()
        shifted[2, 3] += offset_m
        shifted_local_ros = invert_transform(T_base_gripper) @ shifted
        variants.append(
            VariantSpec(
                name=f"{selected_method.lower()}_world_z_{offset_mm:+.0f}mm".replace("+", "p").replace("-", "m"),
                description=f"Selected {selected_method} solution with world-z shift of {offset_mm:+.0f} mm",
                calib_camera_convention="ros",
                T_base_camera=shifted,
                local_pose_opengl=pose_ros_to_opengl(shifted_local_ros),
                source_method=selected_method,
                source_kind="selected_world_z_probe",
                world_z_offset_m=offset_m,
                notes={"offset_axis": "world_z", "offset_mm": float(offset_mm)},
                T_gripper_camera_assumed=shifted_local_ros,
            )
        )

    return variants


def render_variant(
    args: argparse.Namespace,
    sensor_name: str,
    source_run_dir: Path,
    source_result: dict[str, Any],
    reference_ee_pose: np.ndarray,
    variant: VariantSpec,
    work_root: Path,
) -> dict[str, Any]:
    variant_root = ensure_dir(work_root / sanitize_name(variant.name))
    calibration_dir = ensure_dir(variant_root / "calibration")
    output_dir = ensure_dir(variant_root / "render")
    payload = make_minimal_calibration_payload(
        source_result=source_result,
        T_base_camera=variant.T_base_camera,
        reference_ee_pose=reference_ee_pose,
        variant=variant,
        decimals=args.decimals,
    )
    save_json(calibration_dir / "calibration_result.json", to_jsonable(payload))
    shutil.copy2(source_run_dir / "camera_intrinsics.json", calibration_dir / "camera_intrinsics.json")

    container_calibration_dir = map_host_path_to_container(
        calibration_dir, args.host_workspace_root, args.container_workspace_root
    )
    container_output_dir = map_host_path_to_container(
        output_dir, args.host_workspace_root, args.container_workspace_root
    )
    container_arm_txt_path = map_host_path_to_container(
        args.arm_txt_path.expanduser().resolve(), args.host_workspace_root, args.container_workspace_root
    )
    container_arm_npy_path = map_host_path_to_container(
        args.arm_npy_path.expanduser().resolve(), args.host_workspace_root, args.container_workspace_root
    )
    container_real_image_path = map_host_path_to_container(
        real_image_for_arm(source_result["arm_name"]).expanduser().resolve(),
        args.host_workspace_root,
        args.container_workspace_root,
    )

    command = [
        "docker",
        "exec",
        args.container_name,
        args.container_python,
        args.container_render_script,
        "--device",
        args.device,
        "--calib_camera_convention",
        variant.calib_camera_convention,
        "--arm_txt_path",
        container_arm_txt_path,
        "--arm_npy_path",
        container_arm_npy_path,
        "--arm_record_index",
        str(args.arm_record_index),
        "--overlay_alpha",
        str(args.overlay_alpha),
        f"--{sensor_name}_dir",
        container_calibration_dir,
        f"--real_{sensor_name}_image",
        container_real_image_path,
        "--output_dir",
        container_output_dir,
    ]
    if not args.no_headless:
        command.append("--headless")

    print(f"render_variant[{variant.name}]:")
    print(" ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise RuntimeError(f"render failed for variant {variant.name} with exit code {completed.returncode}")

    run_summary_path = output_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    camera_summary = run_summary["cameras"][sensor_name]
    return {
        "variant_name": variant.name,
        "description": variant.description,
        "source_kind": variant.source_kind,
        "source_method": variant.source_method,
        "calib_camera_convention": variant.calib_camera_convention,
        "world_z_offset_m": float(variant.world_z_offset_m),
        "output_dir": str(output_dir),
        "render_summary_path": str(run_summary_path),
        "sim_rgba_path": map_container_path_to_host(
            camera_summary.get("sim_rgba_path"), args.host_workspace_root, args.container_workspace_root
        ),
        "overlay_path": map_container_path_to_host(
            camera_summary.get("overlay_path"), args.host_workspace_root, args.container_workspace_root
        ),
        "visible_mask_path": map_container_path_to_host(
            camera_summary.get("visible_mask_path"), args.host_workspace_root, args.container_workspace_root
        ),
        "real_resized_path": map_container_path_to_host(
            camera_summary.get("real_resized_path"), args.host_workspace_root, args.container_workspace_root
        ),
        "T_world_camera_ros": camera_summary.get("T_world_camera_ros"),
        "T_sensor_pose_opengl": camera_summary.get("T_sensor_pose_opengl"),
        "local_pose_opengl_expected": to_jsonable(variant.local_pose_opengl),
        "notes": to_jsonable(variant.notes or {}),
    }


def try_export_results(work_root: Path, export_dir: Path) -> str | None:
    try:
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.copytree(work_root, export_dir)
        return str(export_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"export_warning: failed to copy diagnostics to {export_dir}: {exc}")
        return None


def main() -> int:
    args = parse_args()

    run_dir = resolve_saved_run_dir(args.run_dir)
    arm_txt_path = ensure_file(args.arm_txt_path.expanduser().resolve())
    arm_npy_path = ensure_file(args.arm_npy_path.expanduser().resolve())
    host_workspace_root = args.host_workspace_root.expanduser().resolve()
    host_container_runs_dir = ensure_dir(args.host_container_runs_dir.expanduser().resolve())
    source_result_path = ensure_file(run_dir / "calibration_result.json")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))

    run_name, arm_name, camera_name, intrinsics, samples = load_saved_run(run_dir)
    sensor_name = camera_sensor_name_for_arm(arm_name)
    arm_payload = load_arm_pose_payload(arm_npy_path)
    selected_record = select_arm_pose_record(arm_payload, args.arm_record_index)
    reference_ee_pose = np.asarray(selected_record[ee_pose_key_for_arm(arm_name)], dtype=np.float64).reshape(6)

    print(f"diagnose_run_dir: {run_dir}")
    print(f"arm_name: {arm_name}")
    print(f"camera_name: cam_{camera_name}")
    print(f"sample_count: {len(samples)}")
    print(f"arm_record_index: {args.arm_record_index}")

    eye_in_hand_result = run_eye_in_hand_calibration(samples)
    print(f"selected_method_from_recompute: {eye_in_hand_result['best_method']}")

    variants = build_variants(
        source_result=source_result,
        arm_name=arm_name,
        selected_record=selected_record,
        eye_in_hand_result=eye_in_hand_result,
        world_z_offsets_mm=args.world_z_offsets_mm,
    )
    print(f"variant_count: {len(variants)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_root = ensure_dir(host_container_runs_dir / f"{run_name}__eye_in_hand_method_diag_{timestamp}")
    render_results: list[dict[str, Any]] = []
    default_local_pose_gl = default_local_pose_opengl_for_sensor(sensor_name)

    for variant in variants:
        delta_transform = invert_transform(default_local_pose_gl) @ variant.local_pose_opengl
        delta_translation_mm = float(np.linalg.norm(delta_transform[:3, 3]) * 1000.0)
        delta_rotation_deg = rotation_error_deg(default_local_pose_gl[:3, :3], variant.local_pose_opengl[:3, :3])
        variant.notes = {
            **(variant.notes or {}),
            "delta_from_default_local_translation_mm": delta_translation_mm,
            "delta_from_default_local_rotation_deg": delta_rotation_deg,
        }
        print(
            f"variant[{variant.name}] delta_from_default_local="
            f"{delta_translation_mm:.2f}mm/{delta_rotation_deg:.2f}deg"
        )
        render_results.append(
            render_variant(
                args=args,
                sensor_name=sensor_name,
                source_run_dir=run_dir,
                source_result=source_result,
                reference_ee_pose=reference_ee_pose,
                variant=variant,
                work_root=work_root,
            )
        )

    sim_contact_sheet = build_contact_sheet(
        title=f"{run_name} sim wrist views",
        image_paths=[
            (result["variant_name"], Path(result["sim_rgba_path"]))
            for result in render_results
            if result.get("sim_rgba_path")
        ],
        output_path=work_root / "sim_contact_sheet.png",
    )
    overlay_contact_sheet = build_contact_sheet(
        title=f"{run_name} overlay wrist views",
        image_paths=[
            (result["variant_name"], Path(result["overlay_path"]))
            for result in render_results
            if result.get("overlay_path")
        ],
        output_path=work_root / "overlay_contact_sheet.png",
    )

    summary_payload = {
        "source_run_dir": str(run_dir),
        "source_calibration_result_path": str(source_result_path),
        "arm_txt_path": str(arm_txt_path),
        "arm_npy_path": str(arm_npy_path),
        "arm_record_index": int(args.arm_record_index),
        "arm_name": arm_name,
        "camera_name": f"cam_{camera_name}",
        "sensor_name": sensor_name,
        "selected_record": to_jsonable(selected_record),
        "selected_method_from_recompute": eye_in_hand_result["best_method"],
        "all_method_metrics_from_recompute": {
            method_name: to_jsonable(method_result)
            for method_name, method_result in eye_in_hand_result["all_results"].items()
        },
        "variants": render_results,
        "sim_contact_sheet_path": None if sim_contact_sheet is None else str(sim_contact_sheet),
        "overlay_contact_sheet_path": None if overlay_contact_sheet is None else str(overlay_contact_sheet),
    }
    summary_path = work_root / "diagnostic_summary.json"
    save_json(summary_path, to_jsonable(summary_payload))
    print(f"diagnostic_summary: {summary_path}")

    export_dir = run_dir / f"{args.export_subdir_prefix}_{timestamp}"
    exported_path = try_export_results(work_root, export_dir)
    if exported_path is not None:
        print(f"exported_copy: {exported_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
