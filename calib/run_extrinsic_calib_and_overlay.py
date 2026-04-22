#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import posixpath
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = PROJECT_ROOT / "calib"
CALIB_SCRIPT = CALIB_DIR / "dual_piper_arm_extrinsic_calib.py"
HOST_RUNS_ROOT = CALIB_DIR / "runs"

DEFAULT_HOST_CONTAINER_RUNS_DIR = Path(
    "/home/edemlab/challenge_ws/embodichain_ws/calib_real_vs_sim/runs"
)
DEFAULT_CONTAINER_NAME = "embodichain"
DEFAULT_CONTAINER_PYTHON = "/root/miniconda3/envs/py310/bin/python"
DEFAULT_CONTAINER_RENDER_SCRIPT = "/root/workspace/calib_real_vs_sim/render_robot_calib_overlay.py"
DEFAULT_CONTAINER_RUNS_DIR = "/root/workspace/calib_real_vs_sim/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dual Piper extrinsic calibration, then render the real-vs-sim overlay "
            "inside the embodichain container when calibration passes quality gates."
        )
    )
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME)
    parser.add_argument("--container-python", default=DEFAULT_CONTAINER_PYTHON)
    parser.add_argument("--container-render-script", default=DEFAULT_CONTAINER_RENDER_SCRIPT)
    parser.add_argument("--container-runs-dir", default=DEFAULT_CONTAINER_RUNS_DIR)
    parser.add_argument(
        "--host-container-runs-dir",
        type=Path,
        default=DEFAULT_HOST_CONTAINER_RUNS_DIR,
        help=(
            "host path that is visible in the container as --container-runs-dir; "
            "default matches /home/edemlab/challenge_ws/embodichain_ws -> /root/workspace"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-headless", action="store_true", help="do not pass --headless to renderer")
    parser.add_argument("--max-reproj-rmse-px", type=float, default=2.0)
    parser.add_argument("--max-cam-trans-rmse-mm", type=float, default=50.0)
    parser.add_argument("--max-cam-rot-rmse-deg", type=float, default=5.0)
    parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="render whenever calibration_result.json exists, without metric thresholds",
    )
    parser.add_argument(
        "calib_args",
        nargs=argparse.REMAINDER,
        help="extra args forwarded to dual_piper_arm_extrinsic_calib.py; prefix them with --",
    )
    return parser.parse_args()


def strip_remainder_separator(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


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


def validate_calibration_result(run_dir: Path, args: argparse.Namespace) -> None:
    result = load_calibration_result(run_dir)
    sample_count = int(result.get("sample_count", 0))
    selected_method = result.get("selected_method")
    if sample_count < 3:
        raise RuntimeError(f"calibration has too few samples: {sample_count}")
    if not selected_method:
        raise RuntimeError("calibration_result.json is missing selected_method")
    if args.no_quality_gate:
        print(f"quality_gate: skipped (selected_method={selected_method}, sample_count={sample_count})")
        return

    reproj = metric_value(result, "reprojection_error_rmse_px")
    cam_trans = metric_value(result, "camera_translation_consistency_rmse_mm")
    cam_rot = metric_value(result, "camera_rotation_consistency_rmse_deg")
    failures = []
    if reproj > args.max_reproj_rmse_px:
        failures.append(f"reproj_rmse_px={reproj:.4f} > {args.max_reproj_rmse_px:.4f}")
    if cam_trans > args.max_cam_trans_rmse_mm:
        failures.append(f"cam_trans_rmse_mm={cam_trans:.4f} > {args.max_cam_trans_rmse_mm:.4f}")
    if cam_rot > args.max_cam_rot_rmse_deg:
        failures.append(f"cam_rot_rmse_deg={cam_rot:.4f} > {args.max_cam_rot_rmse_deg:.4f}")
    if failures:
        raise RuntimeError("calibration quality gate failed: " + "; ".join(failures))
    print(
        "quality_gate: passed "
        f"(method={selected_method}, samples={sample_count}, "
        f"reproj={reproj:.4f}px, cam_trans={cam_trans:.4f}mm, cam_rot={cam_rot:.4f}deg)"
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_run_to_container_visible_dir(run_dir: Path, args: argparse.Namespace) -> tuple[Path, str, bool]:
    host_container_runs_dir = args.host_container_runs_dir.expanduser().resolve()
    host_container_runs_dir.mkdir(parents=True, exist_ok=True)

    run_dir_resolved = run_dir.resolve()
    if is_relative_to(run_dir_resolved, host_container_runs_dir):
        relative_run = run_dir_resolved.relative_to(host_container_runs_dir)
        container_run_dir = posixpath.join(args.container_runs_dir, relative_run.as_posix())
        print(f"container_access: run already under mounted workspace: {run_dir_resolved}")
        return run_dir_resolved, container_run_dir, False

    mirror_run_dir = host_container_runs_dir / run_dir.name
    if mirror_run_dir.exists() or mirror_run_dir.is_symlink():
        remove_path(mirror_run_dir)
    shutil.copytree(run_dir_resolved, mirror_run_dir, symlinks=False)
    container_run_dir = posixpath.join(args.container_runs_dir, run_dir.name)
    print(f"container_access: mirrored {run_dir_resolved} -> {mirror_run_dir}")
    return mirror_run_dir, container_run_dir, True


def copy_run_back(mirror_run_dir: Path, original_run_dir: Path) -> None:
    shutil.copytree(mirror_run_dir, original_run_dir, dirs_exist_ok=True)
    print(f"synced_overlay_outputs: {mirror_run_dir} -> {original_run_dir}")


def run_calibration(calib_args: list[str]) -> Path:
    HOST_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    before_dirs = direct_run_dirs(HOST_RUNS_ROOT)
    start_time = time.time()
    command = [sys.executable, str(CALIB_SCRIPT), *calib_args]
    print("running_calibration:")
    print(" ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    run_dir = newest_run_dir(HOST_RUNS_ROOT, created_after=start_time, previous_dirs=before_dirs)
    print(f"calibration_run_dir: {run_dir}")
    return run_dir


def run_overlay(args: argparse.Namespace, container_run_dir: str) -> None:
    command = [
        "docker",
        "exec",
        args.container_name,
        args.container_python,
        args.container_render_script,
        "--device",
        args.device,
    ]
    if not args.no_headless:
        command.append("--headless")
    command.extend(
        [
            "--cam_high_dir",
            container_run_dir,
            "--output_dir",
            container_run_dir,
        ]
    )
    print("running_overlay:")
    print(" ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    calib_args = strip_remainder_separator(args.calib_args)
    run_dir = run_calibration(calib_args)
    validate_calibration_result(run_dir, args)
    mirror_run_dir, container_run_dir, needs_copy_back = copy_run_to_container_visible_dir(run_dir, args)
    run_overlay(args, container_run_dir)
    if needs_copy_back:
        copy_run_back(mirror_run_dir, run_dir)
    print(f"done: overlay outputs are available under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
