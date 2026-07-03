#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_sort_key(path):
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def collect_images(image_dir):
    image_dir = Path(image_dir)
    image_paths = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    image_paths = sorted(image_paths, key=natural_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return image_paths


def load_pose_json(pose_file):
    pose_path = Path(pose_file)
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    poses = payload.get("poses", payload)
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"No pose records found in {pose_path}")
    return payload, poses


def compute_stride(args):
    if args.stride is not None:
        if args.stride < 1:
            raise ValueError("--stride must be >= 1")
        return args.stride
    if args.source_fps is None or args.target_fps is None:
        raise ValueError("Either --stride or both --source-fps/--target-fps are required")
    if args.source_fps <= 0 or args.target_fps <= 0:
        raise ValueError("--source-fps and --target-fps must be positive")
    stride = max(1, int(round(args.source_fps / args.target_fps)))
    return stride


def link_or_copy(src, dst, copy_files):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def yaw_matrix(yaw, template_matrix=None):
    matrix = np.eye(4, dtype=float)
    if template_matrix is not None:
        matrix = np.array(template_matrix, dtype=float).copy()
    c = math.cos(yaw)
    s = math.sin(yaw)
    matrix[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return matrix


def pose_matrix(record):
    if isinstance(record, dict):
        if "matrix" in record:
            return np.array(record["matrix"], dtype=float)
        if "translation" in record:
            matrix = np.eye(4, dtype=float)
            matrix[:3, 3] = np.array(record["translation"][:3], dtype=float)
            return matrix
    matrix = np.array(record, dtype=float)
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"Unsupported pose record format: {type(record)}")


def infer_source_timestamps(poses, source_fps):
    timestamps = []
    for idx, record in enumerate(poses):
        if isinstance(record, dict) and "timestamp" in record:
            timestamps.append(float(record["timestamp"]))
        elif source_fps:
            timestamps.append(float(idx / source_fps))
        else:
            timestamps.append(float(idx))
    return np.array(timestamps, dtype=float)


def build_stop_aware_motion(poses, source_fps, target_interval, args):
    matrices = [pose_matrix(record) for record in poses]
    positions = np.array([matrix[:3, 3] for matrix in matrices], dtype=float)
    timestamps = infer_source_timestamps(poses, source_fps)
    xy = positions[:, :2]
    n = len(xy)
    window = max(1, int(args.heading_window))

    raw_headings = np.zeros(n, dtype=float)
    speed_mps = np.zeros(n, dtype=float)
    valid_heading = np.zeros(n, dtype=bool)
    for idx in range(n):
        left = max(0, idx - window)
        right = min(n - 1, idx + window)
        if right == left:
            continue
        delta = xy[right] - xy[left]
        dt = max(timestamps[right] - timestamps[left], 1e-9)
        speed = float(np.linalg.norm(delta) / dt)
        speed_mps[idx] = speed
        if speed >= args.stop_speed_threshold:
            raw_headings[idx] = math.atan2(delta[1], delta[0])
            valid_heading[idx] = True

    if not np.any(valid_heading):
        valid_heading[:] = True
        raw_headings[:] = 0.0

    headings = np.unwrap(raw_headings)
    last_heading = float(headings[np.flatnonzero(valid_heading)[0]])
    for idx in range(n):
        if valid_heading[idx]:
            last_heading = float(headings[idx])
        else:
            headings[idx] = last_heading

    next_heading = last_heading
    for idx in range(n - 1, -1, -1):
        if valid_heading[idx]:
            next_heading = float(headings[idx])
        elif idx == 0:
            headings[idx] = next_heading

    speed_per_step = speed_mps * target_interval
    curvatures = np.zeros(n, dtype=float)
    for idx in range(1, n - 1):
        if speed_mps[idx] < args.stop_speed_threshold:
            continue
        dt = max(timestamps[idx + 1] - timestamps[idx - 1], 1e-9)
        distance = max(speed_mps[idx] * dt, args.min_curvature_distance)
        dtheta = headings[idx + 1] - headings[idx - 1]
        curvatures[idx] = float(np.clip(dtheta / distance, -args.max_curvature, args.max_curvature))
    if n > 1:
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]

    velocities = np.zeros((n, 3), dtype=float)
    velocities[:, 0] = speed_per_step * np.cos(headings)
    velocities[:, 1] = speed_per_step * np.sin(headings)

    return {
        "matrices": matrices,
        "heading_rad": headings,
        "speed_mps": speed_mps,
        "speed_per_step": speed_per_step,
        "velocity": velocities,
        "curvature": curvatures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--pose-file", required=True)
    parser.add_argument("--output-image-dir", required=True)
    parser.add_argument("--output-pose-file", required=True)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--source-fps", type=float, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stop-aware-motion", action="store_true")
    parser.add_argument("--stop-speed-threshold", type=float, default=0.05)
    parser.add_argument("--heading-window", type=int, default=15)
    parser.add_argument("--max-curvature", type=float, default=2.0)
    parser.add_argument("--min-curvature-distance", type=float, default=0.05)
    args = parser.parse_args()

    image_paths = collect_images(args.image_dir)
    payload, poses = load_pose_json(args.pose_file)
    usable = min(len(image_paths), len(poses))
    if usable < 1:
        raise ValueError("No usable image/pose pairs")

    stride = compute_stride(args)
    selected_indices = list(range(0, usable, stride))
    if args.max_frames is not None:
        selected_indices = selected_indices[: args.max_frames]
    if not selected_indices:
        raise ValueError("Sampling produced no frames")

    output_image_dir = Path(args.output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_pose_file = Path(args.output_pose_file)
    output_pose_file.parent.mkdir(parents=True, exist_ok=True)

    target_interval = 1.0 / args.target_fps if args.target_fps else None
    motion = None
    if args.stop_aware_motion:
        if target_interval is None:
            raise ValueError("--stop-aware-motion requires --target-fps")
        motion = build_stop_aware_motion(poses[:usable], args.source_fps, target_interval, args)

    sampled_poses = []
    for output_index, source_index in enumerate(selected_indices):
        src_image = image_paths[source_index]
        dst_name = f"{args.name or Path(args.image_dir).name}_{output_index:04d}{src_image.suffix.lower()}"
        dst_image = output_image_dir / dst_name
        link_or_copy(src_image, dst_image, args.copy_images)

        record = poses[source_index]
        if isinstance(record, dict):
            sampled = dict(record)
        else:
            sampled = {"matrix": record}
        if motion is not None:
            matrix = yaw_matrix(
                float(motion["heading_rad"][source_index]),
                template_matrix=motion["matrices"][source_index],
            )
            sampled["matrix"] = matrix.tolist()
            sampled["heading_rad"] = float(motion["heading_rad"][source_index])
            sampled["speed_mps"] = float(motion["speed_mps"][source_index])
            sampled["speed_per_step"] = float(motion["speed_per_step"][source_index])
            sampled["velocity"] = motion["velocity"][source_index].tolist()
            sampled["curvature"] = float(motion["curvature"][source_index])
            sampled["motion_source"] = "stop_aware_30fps"
        sampled["image_index"] = output_index
        sampled["source_index"] = int(source_index)
        sampled["source_image"] = src_image.name
        sampled["image"] = dst_image.name
        if "timestamp" in sampled:
            sampled["source_timestamp"] = sampled["timestamp"]
        if target_interval is not None:
            sampled["timestamp"] = float(output_index * target_interval)
        sampled_poses.append(sampled)

    sampled_payload = {
        "name": args.name or f"{Path(args.pose_file).stem}_sampled",
        "source_image_dir": str(Path(args.image_dir).resolve()),
        "source_pose_file": str(Path(args.pose_file).resolve()),
        "sampling_stride": int(stride),
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "poses": sampled_poses,
    }
    if motion is not None:
        sampled_payload["motion_fields"] = {
            "source": "stop_aware_30fps",
            "stop_speed_threshold_mps": args.stop_speed_threshold,
            "heading_window_frames": args.heading_window,
            "max_curvature": args.max_curvature,
            "min_curvature_distance_m": args.min_curvature_distance,
            "target_interval_s": target_interval,
        }
    if isinstance(payload, dict):
        for key in ("source", "monocular_scale_note", "raw_traj_format"):
            if key in payload:
                sampled_payload[key] = payload[key]

    output_pose_file.write_text(json.dumps(sampled_payload, indent=2), encoding="utf-8")
    summary = {
        "output_image_dir": str(output_image_dir),
        "output_pose_file": str(output_pose_file),
        "input_pairs": int(usable),
        "sampled_pairs": int(len(sampled_poses)),
        "stride": int(stride),
        "stop_aware_motion": bool(motion is not None),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
