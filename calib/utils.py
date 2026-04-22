from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation


HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

DEFAULT_TAG_DICTIONARY = "DICT_APRILTAG_36h11"
DEFAULT_TAG_ID = 0
DEFAULT_TAG_SIZE_MM = 100.0
DEFAULT_TAG_DPI = 300
DEFAULT_TAG_PNG_NAME = "default_tag_a4.png"
DEFAULT_TAG_METADATA_NAME = "default_tag_metadata.json"

# PiperController currently exposes ee_pose as:
# - xyz in meters
# - rxyz as SDK Euler angles scaled down by 1/1000
# The SDK docs define RX/RY/RZ as 0.001 degrees in xyz order, so we need to
# multiply the controller values by 1000 to recover degrees before building a
# rotation matrix.
PIPER_EE_ROTATION_ORDER = "xyz"
PIPER_EE_ROTATION_SCALE_TO_DEGREES = 1000.0


@dataclass
class TagDetection:
    marker_id: int
    corners_px: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    transform_camera_to_tag: np.ndarray


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_tag_config() -> dict[str, Any]:
    return {
        "dictionary_name": DEFAULT_TAG_DICTIONARY,
        "marker_id": DEFAULT_TAG_ID,
        "tag_size_mm": float(DEFAULT_TAG_SIZE_MM),
        "dpi": int(DEFAULT_TAG_DPI),
        "png_name": DEFAULT_TAG_PNG_NAME,
        "metadata_name": DEFAULT_TAG_METADATA_NAME,
    }


def timestamp_for_path(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def normalize_arm_label(text: str) -> str:
    value = text.strip().lower()
    if value == "l":
        return "left_arm"
    if value == "r":
        return "right_arm"
    raise ValueError("arm must be 'l' or 'r'")


def arm_to_can_bus(arm_name: str) -> str:
    if arm_name == "left_arm":
        return "can0"
    if arm_name == "right_arm":
        return "can1"
    raise ValueError(f"unsupported arm name: {arm_name}")


def pose6d_to_matrix(pose: Iterable[float]) -> np.ndarray:
    pose_array = np.asarray(list(pose), dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    rotation_degrees = pose_array[3:] * PIPER_EE_ROTATION_SCALE_TO_DEGREES
    transform[:3, :3] = Rotation.from_euler(
        PIPER_EE_ROTATION_ORDER,
        rotation_degrees,
        degrees=True,
    ).as_matrix()
    transform[:3, 3] = pose_array[:3]
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def make_transform(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_vector, dtype=np.float64).reshape(3)
    return transform


def sanitize_rotation_matrix(
    rotation_matrix: np.ndarray,
    *,
    context: str,
    orthogonality_tol: float = 1e-3,
) -> np.ndarray:
    matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    determinant = float(np.linalg.det(matrix))
    orthogonality_error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
    if determinant <= 0.0:
        raise ValueError(
            f"{context} returned a left-handed rotation matrix "
            f"(det={determinant:.6f}, orthogonality_error={orthogonality_error:.3e})"
        )
    if orthogonality_error > orthogonality_tol:
        raise ValueError(
            f"{context} returned a non-orthonormal rotation matrix "
            f"(det={determinant:.6f}, orthogonality_error={orthogonality_error:.3e})"
        )

    # Project onto SO(3) to remove small numeric drift from downstream solvers.
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError(f"{context} could not be projected to a proper rotation matrix")
    return rotation


def rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    rotation = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3))
    return float(rotation.magnitude() * 180.0 / math.pi)


def average_rotation_matrix(rotation_matrices: list[np.ndarray]) -> np.ndarray:
    matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if len(matrices) == 1:
        return matrices[0]
    quaternions = Rotation.from_matrix(matrices).as_quat()
    reference = quaternions[0]
    for idx in range(1, len(quaternions)):
        if float(np.dot(quaternions[idx], reference)) < 0.0:
            quaternions[idx] *= -1.0
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quat in quaternions:
        accumulator += np.outer(quat, quat)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    mean_quaternion = eigenvectors[:, np.argmax(eigenvalues)]
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    return Rotation.from_quat(mean_quaternion).as_matrix()


def average_transform(transforms: list[np.ndarray]) -> np.ndarray:
    transform_stack = np.asarray(transforms, dtype=np.float64)
    rotation = average_rotation_matrix([transform[:3, :3] for transform in transform_stack])
    translation = np.mean(transform_stack[:, :3, 3], axis=0)
    return make_transform(rotation, translation)


def matrix_to_rounded_nested_list(matrix: np.ndarray, decimals: int = 6) -> list[list[float]]:
    rounded = np.round(np.asarray(matrix, dtype=np.float64), decimals=decimals)
    result = []
    for row in rounded:
        result.append([float(value) for value in row])
    return result


def format_matrix(matrix: np.ndarray, decimals: int = 6, indent: int = 2) -> str:
    rounded = np.asarray(matrix_to_rounded_nested_list(matrix, decimals=decimals), dtype=np.float64)
    lines = ["["]
    pad = " " * indent
    for row_idx, row in enumerate(rounded):
        row_text = ", ".join(f"{float(value):.{decimals}f}" for value in row)
        suffix = "," if row_idx < len(rounded) - 1 else ""
        lines.append(f"{pad}[{row_text}]{suffix}")
    lines.append("]")
    return "\n".join(lines)


def format_matrix_list(matrices: list[np.ndarray], decimals: int = 6, indent: int = 2) -> str:
    if not matrices:
        return "[]"
    pad = " " * indent
    lines = ["["]
    for idx, matrix in enumerate(matrices):
        formatted = format_matrix(matrix, decimals=decimals, indent=indent + 2).splitlines()
        if idx < len(matrices) - 1:
            formatted[-1] += ","
        lines.extend([pad + line for line in formatted])
    lines.append("]")
    return "\n".join(lines)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def overlay_status_panel(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    image = image_bgr.copy()
    if not lines:
        return image
    line_height = 24
    margin = 10
    panel_height = margin * 2 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0.0)
    for idx, line in enumerate(lines):
        y = margin + (idx + 1) * line_height - 6
        cv2.putText(
            image,
            line,
            (margin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def aruco_dictionary_from_name(dictionary_name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"unsupported cv2.aruco dictionary: {dictionary_name}")
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def marker_object_points(tag_size_m: float) -> np.ndarray:
    half_size = tag_size_m / 2.0
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float64,
    )


def create_a4_tag_png(
    output_path: Path,
    dictionary_name: str,
    marker_id: int,
    tag_size_mm: float,
    dpi: int = 300,
) -> dict[str, Any]:
    if tag_size_mm <= 0:
        raise ValueError("tag size must be positive")

    dictionary = aruco_dictionary_from_name(dictionary_name)
    a4_width_px = int(round(210.0 / 25.4 * dpi))
    a4_height_px = int(round(297.0 / 25.4 * dpi))
    margin_px = int(round(12.0 / 25.4 * dpi))
    text_band_px = int(round(18.0 / 25.4 * dpi))
    marker_px = int(round(tag_size_mm / 25.4 * dpi))
    max_marker_px = min(a4_width_px - margin_px * 2, a4_height_px - margin_px * 2 - text_band_px)
    if marker_px > max_marker_px:
        raise ValueError(
            f"tag size {tag_size_mm:.1f} mm does not fit on A4 at {dpi} dpi with margins"
        )

    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
    canvas = np.full((a4_height_px, a4_width_px), 255, dtype=np.uint8)
    start_x = (a4_width_px - marker_px) // 2
    start_y = (a4_height_px - marker_px - text_band_px) // 2
    canvas[start_y : start_y + marker_px, start_x : start_x + marker_px] = marker_image

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        canvas_bgr,
        (start_x, start_y),
        (start_x + marker_px, start_y + marker_px),
        (0, 0, 0),
        2,
    )

    lines = [
        f"{dictionary_name}  id={marker_id}",
        f"marker={tag_size_mm:.1f}mm  paper=A4  dpi={dpi}",
    ]
    for idx, line in enumerate(lines):
        y = start_y + marker_px + 42 + idx * 36
        cv2.putText(
            canvas_bgr,
            line,
            (margin_px, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas_bgr)
    return {
        "output_path": str(output_path),
        "dictionary_name": dictionary_name,
        "marker_id": marker_id,
        "tag_size_mm": float(tag_size_mm),
        "dpi": int(dpi),
        "a4_size_px": [a4_width_px, a4_height_px],
        "marker_size_px": int(marker_px),
    }


def get_realsense_intrinsics(sensor: Any) -> dict[str, Any]:
    profile = sensor.pipeline.get_active_profile()
    stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = stream_profile.get_intrinsics()
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intrinsics.coeffs[:5], dtype=np.float64)
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "coeffs": [float(value) for value in dist_coeffs],
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
    }


def detect_tag_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    dictionary_name: str,
    marker_id: int,
    tag_size_m: float,
) -> tuple[TagDetection | None, np.ndarray]:
    dictionary = aruco_dictionary_from_name(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners_list, ids, _ = detector.detectMarkers(image_bgr)

    annotated = image_bgr.copy()
    if ids is None or len(ids) == 0:
        return None, annotated

    cv2.aruco.drawDetectedMarkers(annotated, corners_list, ids)
    flat_ids = ids.reshape(-1).tolist()
    if marker_id not in flat_ids:
        return None, annotated

    marker_index = flat_ids.index(marker_id)
    marker_corners = np.asarray(corners_list[marker_index], dtype=np.float64).reshape(4, 2)
    object_points = marker_object_points(tag_size_m)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        marker_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, annotated

    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    reprojection_error_px = float(
        np.sqrt(np.mean(np.sum((projected_points - marker_corners) ** 2, axis=1)))
    )

    cv2.drawFrameAxes(
        annotated,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        tag_size_m * 0.5,
        2,
    )

    transform_camera_to_tag = make_transform(
        Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix(),
        np.asarray(tvec, dtype=np.float64).reshape(3),
    )

    cv2.putText(
        annotated,
        f"tag={marker_id}  reproj={reprojection_error_px:.3f}px",
        (10, annotated.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    detection = TagDetection(
        marker_id=marker_id,
        corners_px=marker_corners,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        reprojection_error_px=reprojection_error_px,
        transform_camera_to_tag=transform_camera_to_tag,
    )
    return detection, annotated


def transform_error(reference_transform: np.ndarray, query_transform: np.ndarray) -> tuple[float, float]:
    delta = invert_transform(reference_transform) @ query_transform
    translation_error_m = float(np.linalg.norm(delta[:3, 3]))
    rotation_error_deg = rotation_angle_deg(delta[:3, :3])
    return translation_error_m, rotation_error_deg


def sample_motion_summary(base_to_gripper_list: list[np.ndarray]) -> dict[str, Any]:
    if not base_to_gripper_list:
        return {
            "translation_span_xyz_mm": [0.0, 0.0, 0.0],
            "translation_span_norm_mm": 0.0,
            "max_relative_rotation_deg": 0.0,
        }
    translations = np.stack([transform[:3, 3] for transform in base_to_gripper_list], axis=0)
    span_xyz_mm = (translations.max(axis=0) - translations.min(axis=0)) * 1000.0
    reference_rotation = base_to_gripper_list[0][:3, :3]
    relative_rotation_deg = []
    for transform in base_to_gripper_list:
        relative_rotation_deg.append(
            rotation_angle_deg(reference_rotation.T @ transform[:3, :3])
        )
    return {
        "translation_span_xyz_mm": [float(value) for value in span_xyz_mm],
        "translation_span_norm_mm": float(np.linalg.norm(span_xyz_mm)),
        "max_relative_rotation_deg": float(max(relative_rotation_deg)),
    }


def evaluate_hand_eye_solution(
    samples: list[dict[str, Any]],
    base_to_camera_transform: np.ndarray,
) -> dict[str, Any]:
    base_to_gripper_list = [np.asarray(sample["T_base_gripper"], dtype=np.float64) for sample in samples]
    camera_to_tag_list = [np.asarray(sample["T_camera_tag"], dtype=np.float64) for sample in samples]
    gripper_to_base_list = [invert_transform(transform) for transform in base_to_gripper_list]

    gripper_to_tag_samples = []
    for gripper_to_base, camera_to_tag in zip(gripper_to_base_list, camera_to_tag_list):
        gripper_to_tag_samples.append(gripper_to_base @ base_to_camera_transform @ camera_to_tag)
    gripper_to_tag_mean = average_transform(gripper_to_tag_samples)

    tag_translation_errors_mm = []
    tag_rotation_errors_deg = []
    camera_translation_errors_mm = []
    camera_rotation_errors_deg = []
    per_sample_base_to_camera = []

    for base_to_gripper, camera_to_tag, gripper_to_tag in zip(
        base_to_gripper_list,
        camera_to_tag_list,
        gripper_to_tag_samples,
    ):
        tag_translation_m, tag_rotation_deg = transform_error(gripper_to_tag_mean, gripper_to_tag)
        tag_translation_errors_mm.append(tag_translation_m * 1000.0)
        tag_rotation_errors_deg.append(tag_rotation_deg)

        current_base_to_camera = base_to_gripper @ gripper_to_tag_mean @ invert_transform(camera_to_tag)
        per_sample_base_to_camera.append(current_base_to_camera)
        camera_translation_m, camera_rotation_deg = transform_error(
            base_to_camera_transform,
            current_base_to_camera,
        )
        camera_translation_errors_mm.append(camera_translation_m * 1000.0)
        camera_rotation_errors_deg.append(camera_rotation_deg)

    reprojection_errors_px = [float(sample["reprojection_error_px"]) for sample in samples]
    motion_summary = sample_motion_summary(base_to_gripper_list)

    return {
        "T_base_camera": base_to_camera_transform,
        "T_camera_base": invert_transform(base_to_camera_transform),
        "T_gripper_tag": gripper_to_tag_mean,
        "T_tag_gripper": invert_transform(gripper_to_tag_mean),
        "T_base_camera_per_sample": per_sample_base_to_camera,
        "tag_in_gripper_translation_rmse_mm": float(
            np.sqrt(np.mean(np.square(tag_translation_errors_mm)))
        ),
        "tag_in_gripper_translation_max_mm": float(np.max(tag_translation_errors_mm)),
        "tag_in_gripper_rotation_rmse_deg": float(
            np.sqrt(np.mean(np.square(tag_rotation_errors_deg)))
        ),
        "tag_in_gripper_rotation_max_deg": float(np.max(tag_rotation_errors_deg)),
        "camera_translation_consistency_rmse_mm": float(
            np.sqrt(np.mean(np.square(camera_translation_errors_mm)))
        ),
        "camera_translation_consistency_max_mm": float(np.max(camera_translation_errors_mm)),
        "camera_rotation_consistency_rmse_deg": float(
            np.sqrt(np.mean(np.square(camera_rotation_errors_deg)))
        ),
        "camera_rotation_consistency_max_deg": float(np.max(camera_rotation_errors_deg)),
        "reprojection_error_mean_px": float(np.mean(reprojection_errors_px)),
        "reprojection_error_rmse_px": float(np.sqrt(np.mean(np.square(reprojection_errors_px)))),
        "reprojection_error_max_px": float(np.max(reprojection_errors_px)),
        "motion_summary": motion_summary,
    }


def run_hand_eye_calibration(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 3:
        raise ValueError("at least 3 valid samples are required for hand-eye calibration")

    base_to_gripper_list = [np.asarray(sample["T_base_gripper"], dtype=np.float64) for sample in samples]
    camera_to_tag_list = [np.asarray(sample["T_camera_tag"], dtype=np.float64) for sample in samples]
    gripper_to_base_list = [invert_transform(transform) for transform in base_to_gripper_list]

    results: dict[str, dict[str, Any]] = {}
    for method_name, method_id in HAND_EYE_METHODS.items():
        try:
            rotation_out, translation_out = cv2.calibrateHandEye(
                [transform[:3, :3] for transform in gripper_to_base_list],
                [transform[:3, 3] for transform in gripper_to_base_list],
                [transform[:3, :3] for transform in camera_to_tag_list],
                [transform[:3, 3] for transform in camera_to_tag_list],
                method=method_id,
            )
            rotation_out = sanitize_rotation_matrix(
                rotation_out,
                context=f"cv2.calibrateHandEye({method_name})",
            )
            base_to_camera_transform = make_transform(rotation_out, translation_out)
            metrics = evaluate_hand_eye_solution(samples, base_to_camera_transform)
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            results[method_name] = {"success": False, "error": str(exc)}
            continue
        metrics["success"] = True
        metrics["selection_score"] = (
            metrics["camera_translation_consistency_rmse_mm"]
            + metrics["camera_rotation_consistency_rmse_deg"]
        )
        results[method_name] = metrics

    successful_methods = [
        (method_name, result)
        for method_name, result in results.items()
        if result.get("success")
    ]
    if not successful_methods:
        failure_text = "\n".join(
            f"- {method_name}: {result.get('error', 'unknown error')}"
            for method_name, result in results.items()
        )
        raise RuntimeError(f"all hand-eye calibration methods failed:\n{failure_text}")

    best_method_name, best_result = min(
        successful_methods,
        key=lambda item: (
            float(item[1]["selection_score"]),
            float(item[1]["camera_translation_consistency_rmse_mm"]),
            float(item[1]["camera_rotation_consistency_rmse_deg"]),
        ),
    )
    return {
        "best_method": best_method_name,
        "best_result": best_result,
        "all_results": results,
    }


def evaluate_eye_in_hand_solution(
    samples: list[dict[str, Any]],
    gripper_to_camera_transform: np.ndarray,
) -> dict[str, Any]:
    base_to_gripper_list = [np.asarray(sample["T_base_gripper"], dtype=np.float64) for sample in samples]
    camera_to_tag_list = [np.asarray(sample["T_camera_tag"], dtype=np.float64) for sample in samples]
    base_to_tag_samples = []

    for base_to_gripper, camera_to_tag in zip(base_to_gripper_list, camera_to_tag_list):
        base_to_tag = base_to_gripper @ gripper_to_camera_transform @ camera_to_tag
        base_to_tag_samples.append(base_to_tag)

    base_to_tag_mean = average_transform(base_to_tag_samples)

    tag_translation_errors_mm = []
    tag_rotation_errors_deg = []
    camera_translation_errors_mm = []
    camera_rotation_errors_deg = []
    per_sample_gripper_to_camera = []
    per_sample_base_to_camera = []

    for base_to_gripper, camera_to_tag, base_to_tag in zip(
        base_to_gripper_list,
        camera_to_tag_list,
        base_to_tag_samples,
    ):
        tag_translation_m, tag_rotation_deg = transform_error(base_to_tag_mean, base_to_tag)
        tag_translation_errors_mm.append(tag_translation_m * 1000.0)
        tag_rotation_errors_deg.append(tag_rotation_deg)

        current_gripper_to_camera = (
            invert_transform(base_to_gripper)
            @ base_to_tag_mean
            @ invert_transform(camera_to_tag)
        )
        per_sample_gripper_to_camera.append(current_gripper_to_camera)
        per_sample_base_to_camera.append(base_to_gripper @ current_gripper_to_camera)
        camera_translation_m, camera_rotation_deg = transform_error(
            gripper_to_camera_transform,
            current_gripper_to_camera,
        )
        camera_translation_errors_mm.append(camera_translation_m * 1000.0)
        camera_rotation_errors_deg.append(camera_rotation_deg)

    reprojection_errors_px = [float(sample["reprojection_error_px"]) for sample in samples]
    motion_summary = sample_motion_summary(base_to_gripper_list)

    return {
        "T_gripper_camera": gripper_to_camera_transform,
        "T_camera_gripper": invert_transform(gripper_to_camera_transform),
        "T_base_tag": base_to_tag_mean,
        "T_tag_base": invert_transform(base_to_tag_mean),
        "T_gripper_camera_per_sample": per_sample_gripper_to_camera,
        "T_base_camera_per_sample": per_sample_base_to_camera,
        "camera_in_gripper_translation_rmse_mm": float(
            np.sqrt(np.mean(np.square(camera_translation_errors_mm)))
        ),
        "camera_in_gripper_translation_max_mm": float(np.max(camera_translation_errors_mm)),
        "camera_in_gripper_rotation_rmse_deg": float(
            np.sqrt(np.mean(np.square(camera_rotation_errors_deg)))
        ),
        "camera_in_gripper_rotation_max_deg": float(np.max(camera_rotation_errors_deg)),
        "tag_in_base_translation_rmse_mm": float(
            np.sqrt(np.mean(np.square(tag_translation_errors_mm)))
        ),
        "tag_in_base_translation_max_mm": float(np.max(tag_translation_errors_mm)),
        "tag_in_base_rotation_rmse_deg": float(
            np.sqrt(np.mean(np.square(tag_rotation_errors_deg)))
        ),
        "tag_in_base_rotation_max_deg": float(np.max(tag_rotation_errors_deg)),
        "reprojection_error_mean_px": float(np.mean(reprojection_errors_px)),
        "reprojection_error_rmse_px": float(np.sqrt(np.mean(np.square(reprojection_errors_px)))),
        "reprojection_error_max_px": float(np.max(reprojection_errors_px)),
        "motion_summary": motion_summary,
    }


def run_eye_in_hand_calibration(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 3:
        raise ValueError("at least 3 valid samples are required for eye-in-hand calibration")

    base_to_gripper_list = [np.asarray(sample["T_base_gripper"], dtype=np.float64) for sample in samples]
    camera_to_tag_list = [np.asarray(sample["T_camera_tag"], dtype=np.float64) for sample in samples]

    results: dict[str, dict[str, Any]] = {}
    for method_name, method_id in HAND_EYE_METHODS.items():
        try:
            rotation_out, translation_out = cv2.calibrateHandEye(
                [transform[:3, :3] for transform in base_to_gripper_list],
                [transform[:3, 3] for transform in base_to_gripper_list],
                [transform[:3, :3] for transform in camera_to_tag_list],
                [transform[:3, 3] for transform in camera_to_tag_list],
                method=method_id,
            )
            rotation_out = sanitize_rotation_matrix(
                rotation_out,
                context=f"cv2.calibrateHandEye({method_name})",
            )
            gripper_to_camera_transform = make_transform(rotation_out, translation_out)
            metrics = evaluate_eye_in_hand_solution(samples, gripper_to_camera_transform)
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            results[method_name] = {"success": False, "error": str(exc)}
            continue
        metrics["success"] = True
        metrics["selection_score"] = (
            metrics["tag_in_base_translation_rmse_mm"]
            + metrics["tag_in_base_rotation_rmse_deg"]
        )
        results[method_name] = metrics

    successful_methods = [
        (method_name, result)
        for method_name, result in results.items()
        if result.get("success")
    ]
    if not successful_methods:
        failure_text = "\n".join(
            f"- {method_name}: {result.get('error', 'unknown error')}"
            for method_name, result in results.items()
        )
        raise RuntimeError(f"all eye-in-hand calibration methods failed:\n{failure_text}")

    best_method_name, best_result = min(
        successful_methods,
        key=lambda item: (
            float(item[1]["selection_score"]),
            float(item[1]["tag_in_base_translation_rmse_mm"]),
            float(item[1]["tag_in_base_rotation_rmse_deg"]),
        ),
    )
    return {
        "best_method": best_method_name,
        "best_result": best_result,
        "all_results": results,
    }
