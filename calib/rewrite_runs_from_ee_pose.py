#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calib.dual_piper_arm_extrinsic_calib import finalize_calibration_run, load_saved_run, save_samples_json
from calib.utils import pose6d_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite calibration runs so T_base_gripper is regenerated from ee_pose "
            "using the current calib.utils pose convention, then recompute calibration outputs."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(PROJECT_ROOT / "calib" / "runs"),
        help="directory that contains run subdirectories",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="optional single run directory name to rewrite; default rewrites every immediate child directory",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="decimal places for regenerated samples.json and calibration outputs",
    )
    return parser.parse_args()


def rewrite_sample(sample: dict[str, Any]) -> dict[str, Any]:
    rewritten = dict(sample)
    ee_pose = np.asarray(sample["ee_pose"], dtype=np.float64).reshape(6)
    rewritten["ee_pose"] = ee_pose
    rewritten["T_base_gripper"] = pose6d_to_matrix(ee_pose)
    rewritten["T_camera_tag"] = np.asarray(sample["T_camera_tag"], dtype=np.float64).reshape(4, 4)
    rewritten["tag_corners_px"] = np.asarray(sample["tag_corners_px"], dtype=np.float64).reshape(-1, 2)

    if "arm_ee_pose_snapshot" in sample:
        rewritten["arm_ee_pose_snapshot"] = {
            arm_name: np.asarray(arm_pose, dtype=np.float64).reshape(6)
            for arm_name, arm_pose in sample["arm_ee_pose_snapshot"].items()
        }
    else:
        rewritten["arm_ee_pose_snapshot"] = {
            str(sample["arm_name"]): ee_pose.copy(),
        }

    return rewritten


def iter_run_dirs(runs_root: Path, run_name: str | None) -> list[Path]:
    if run_name is not None:
        run_dir = runs_root / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory not found: {run_dir}")
        return [run_dir]

    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def rewrite_run(run_dir: Path, decimals: int) -> dict[str, Any]:
    samples_path = run_dir / "samples.json"
    intrinsics_path = run_dir / "camera_intrinsics.json"
    if not samples_path.is_file():
        return {
            "status": "skipped",
            "run_name": run_dir.name,
            "reason": f"missing samples file: {samples_path}",
        }
    if not intrinsics_path.is_file():
        return {
            "status": "skipped",
            "run_name": run_dir.name,
            "reason": f"missing intrinsics file: {intrinsics_path}",
        }

    run_name, arm_name, camera_name, intrinsics, samples = load_saved_run(run_dir)
    rewritten_samples = [rewrite_sample(sample) for sample in samples]
    save_samples_json(run_dir, rewritten_samples, decimals=decimals)
    if len(rewritten_samples) < 3:
        error_text = (
            f"有效样本不足，当前仅 {len(rewritten_samples)} 个。"
            "至少需要 3 个不同位姿样本才能计算外参。"
        )
        (run_dir / "calibration_report.txt").write_text(error_text + "\n", encoding="utf-8")
        return {
            "status": "skipped",
            "run_name": run_name,
            "sample_count": len(rewritten_samples),
            "reason": error_text,
        }

    finalize_calibration_run(
        run_dir=run_dir,
        run_name=run_name,
        arm_name=arm_name,
        camera_name=camera_name,
        intrinsics=intrinsics,
        samples=rewritten_samples,
        decimals=decimals,
    )
    return {
        "status": "rewritten",
        "run_name": run_name,
        "sample_count": len(rewritten_samples),
    }


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    run_dirs = iter_run_dirs(runs_root, args.run_name)
    if not run_dirs:
        print(f"no run directories found under: {runs_root}")
        return 0

    print(f"runs_root: {runs_root}")
    print(f"run_count: {len(run_dirs)}")

    rewritten = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []
    for run_dir in run_dirs:
        print("")
        print(f"[rewrite] {run_dir.name}")
        try:
            result = rewrite_run(run_dir, decimals=args.decimals)
        except Exception as exc:
            failures.append((run_dir, str(exc)))
            print(f"[failed] {run_dir.name}: {exc}")
            continue

        if result["status"] == "rewritten":
            rewritten += 1
            print(f"[ok] {result['run_name']} sample_count={result['sample_count']}")
        else:
            skipped += 1
            print(f"[skip] {result['run_name']}: {result['reason']}")

    print("")
    print(f"rewritten: {rewritten}")
    print(f"skipped: {skipped}")
    print(f"failed: {len(failures)}")
    for run_dir, error_text in failures:
        print(f"- {run_dir.name}: {error_text}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
